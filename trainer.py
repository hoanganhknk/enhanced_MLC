# trainer.py
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import DataIterator, AverageMeter, to_device, accuracy, save_checkpoint
from meta import soft_cross_entropy, teacher_ce_loss, teacher_bce_loss, compute_teacher_meta_grads

class Trainer:
    def __init__(self, student, teacher, opt_student, opt_teacher,
                 loader_s, loader_g_train, loader_g_meta, loader_test,
                 args, device, num_classes):
        self.student = student
        self.teacher = teacher
        self.opt_s = opt_student
        self.opt_t = opt_teacher

        self.loader_s = loader_s
        self.g_train_it = DataIterator(loader_g_train)
        self.g_meta_it  = DataIterator(loader_g_meta)
        self.loader_test = loader_test

        self.args = args
        self.device = device
        self.num_classes = num_classes

        self.best_val = -1.0
        self.global_step = 0

        self.buffer = []  # store last k steps {w, x, y}

    @torch.no_grad()
    def evaluate(self, loader):
        self.student.eval()
        accm = AverageMeter()
        for x, y in loader:
            x, y = to_device(x, self.device), to_device(y, self.device)
            logits = self.student(x)
            accm.update(accuracy(logits, y), n=x.size(0))
        return accm.avg

    def train(self):
        self.student.to(self.device)
        self.teacher.to(self.device)

        for epoch in range(self.args.epochs):
            self.student.train()
            self.teacher.train()

            loss_s_m = AverageMeter()
            loss_g_m = AverageMeter()
            loss_t_m = AverageMeter()

            pbar = tqdm(self.loader_s, desc=f"Epoch {epoch+1}/{self.args.epochs}", leave=False)
            for x_s, y_s in pbar:
                self.global_step += 1

                x_s = to_device(x_s, self.device)
                y_s = to_device(y_s, self.device)

                # gold batches
                x_g, y_g = next(self.g_train_it)
                x_g, y_g = to_device(x_g, self.device), to_device(y_g, self.device)

                # ---- store step BEFORE updating student (w^t) + (x_s, y_s) ----
                if self.args.k > 0:
                    self.buffer.append({
                        "w": copy.deepcopy({k: v.detach().cpu() for k, v in self.student.state_dict().items()}),
                        "x": x_s.detach().cpu(),
                        "y": y_s.detach().cpu(),
                    })
                    # keep only last k
                    if len(self.buffer) > self.args.k:
                        self.buffer.pop(0)

                # ---- student update ----
                with torch.no_grad():
                    q_s, _, _, _ = self.teacher.corrected_targets(x_s, y_s, temperature=self.args.temperature)

                logits_s = self.student(x_s)
                logits_g = self.student(x_g)

                loss_s = soft_cross_entropy(logits_s, q_s)
                loss_g = F.cross_entropy(logits_g, y_g)
                loss_total = loss_s + self.args.lambda_gold * loss_g

                self.opt_s.zero_grad(set_to_none=True)
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.args.clip_grad)
                self.opt_s.step()

                loss_s_m.update(loss_s.item(), n=x_s.size(0))
                loss_g_m.update(loss_g.item(), n=x_g.size(0))

                # ---- teacher update every meta_interval steps (default = k) ----
                do_teacher = (self.args.meta_interval > 0) and (self.global_step % self.args.meta_interval == 0)
                if do_teacher and len(self.buffer) > 0:
                    x_m, y_m = next(self.g_meta_it)
                    x_m, y_m = to_device(x_m, self.device), to_device(y_m, self.device)

                    meta_grads, meta_loss_val = compute_teacher_meta_grads(
                        student=self.student,
                        teacher=self.teacher,
                        x_meta=x_m,
                        y_meta=y_m,
                        stored_steps=self.buffer,
                        eta_w=self.args.lr,                  # student lr
                        gamma_w=1.0 - self.args.lr,          # per paper γw=1-ηw :contentReference[oaicite:9]{index=9}
                        temperature=self.args.temperature,
                        device=self.device
                    )

                    # teacher supervised objectives (Eq.12): CE + BCE + META :contentReference[oaicite:10]{index=10}
                    ce = teacher_ce_loss(self.teacher, x_g, y_g)
                    bce = teacher_bce_loss(self.teacher, x_g, y_g, strategy=self.args.teacher_bce)
                    sup_loss = self.args.w_ce * ce + self.args.w_bce * bce

                    self.opt_t.zero_grad(set_to_none=True)
                    sup_loss.backward()

                    # add meta grads to .grad
                    for p, mg in zip(self.teacher.parameters(), meta_grads):
                        if p.grad is None:
                            p.grad = mg.detach()
                        else:
                            p.grad.add_(mg.detach())

                    torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), self.args.clip_grad)
                    self.opt_t.step()

                    loss_t_m.update((sup_loss.item() + meta_loss_val), n=1)

                    # IMPORTANT: clear buffer after teacher update to satisfy "teacher fixed for last k steps"
                    self.buffer.clear()

                pbar.set_postfix({
                    "Ls": f"{loss_s_m.avg:.3f}",
                    "Lg": f"{loss_g_m.avg:.3f}",
                    "Lt": f"{loss_t_m.avg:.3f}",
                })

            # validate on gold_meta
            val_acc = self.evaluate(self.g_meta_it.loader)
            test_acc = self.evaluate(self.loader_test)

            if val_acc > self.best_val:
                self.best_val = val_acc
                save_checkpoint(f"{self.args.out_dir}/best.pth", {
                    "epoch": epoch,
                    "student": self.student.state_dict(),
                    "teacher": self.teacher.state_dict(),
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "args": vars(self.args),
                })

            print(f"[Epoch {epoch+1}] val_acc={val_acc:.4f} test_acc={test_acc:.4f} best_val={self.best_val:.4f}")
