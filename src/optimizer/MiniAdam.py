import copy
import re

import torch


def classify_param_group(name: str) -> str:
    n = name.lower()
    if "q_proj" in n or ".q." in n:                                      return "q_proj"
    if "k_proj" in n or ".k." in n:                                      return "k_proj"
    if "v_proj" in n or ".v." in n:                                      return "v_proj"
    if "o_proj" in n or "out_proj" in n:                                 return "o_proj"
    if any(x in n for x in ("attn", "attention", "self_attn")):          return "attention"
    if any(x in n for x in ("mlp", "ffn", "feed_forward", "fc1", "fc2",
                             "gate_proj", "up_proj", "down_proj")):      return "mlp"
    if "embed" in n:                                                      return "embedding"
    if "norm" in n or "ln" in n:                                         return "norm"
    if "lm_head" in n or "head" in n:                                    return "head"
    return "linear"


def build_param_name(name: str) -> str:
    group = classify_param_group(name)
    m = re.search(r'\.(\d+)\.', name)
    layer_idx = m.group(1) if m else "?"
    return f"{group}/layer{layer_idx}"


class MiniAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = True,
        update_gap: int = 200,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if update_gap <= 0:
            raise ValueError(f"Invalid update_gap value: {update_gap}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "decoupled_weight_decay": decoupled_weight_decay,
            "update_gap": update_gap,
            "projector": None,
            "param_name": None,
            "experiment": None,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                if param.grad.is_sparse:
                    raise RuntimeError("MiniAdam does not support sparse gradients")

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros(
                        (), device=param.device, dtype=torch.float32
                    )

                state["step"] += 1

                if group["weight_decay"] != 0.0:
                    if group["decoupled_weight_decay"]:
                        param.mul_(1.0 - group["lr"] * group["weight_decay"])
                    else:
                        grad = grad.add(param, alpha=group["weight_decay"])

                grad_for_second_moment = grad.float()
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2).add_(
                    grad_for_second_moment.pow(2).mean(), alpha=1.0 - beta2
                )

                projector = self._get_projector(state, group)
                use_projector = projector is not None and grad.dim() == 2

                if use_projector:
                    update = self._low_rank_update(param, grad, state, group, beta1, projector)
                else:
                    update = self._full_rank_update(param, grad, state, beta1)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])
                update = update.div(bias_correction1).div(denom.to(update.dtype))

                param.add_(update, alpha=-group["lr"])

        return loss

    def _full_rank_update(self, param, grad, state, beta1):
        if "exp_avg" not in state or state["exp_avg"].shape != param.shape:
            state["exp_avg"] = torch.zeros_like(param)

        exp_avg = state["exp_avg"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        return exp_avg

    def _low_rank_update(self, param, grad, state, group, beta1, projector):
        needs_update = (
            getattr(projector, "P", None) is None
            or state["step"] == 1
            or state["step"] % group["update_gap"] == 0
        )
        if needs_update:
            projector.update_basis(
                grad,
                param_name=group.get("param_name"),
                step=state["step"],
                experiment=group.get("experiment"),
            )
            state.pop("exp_avg", None)

        low_rank_grad = projector.project(grad)
        if getattr(projector, "was_switched", False):
            state.pop("exp_avg", None)

        if "exp_avg" not in state or state["exp_avg"].shape != low_rank_grad.shape:
            state["exp_avg"] = torch.zeros_like(low_rank_grad)

        exp_avg = state["exp_avg"]
        exp_avg.mul_(beta1).add_(low_rank_grad, alpha=1.0 - beta1)
        return projector.reconstruct(exp_avg).to(param.dtype)

    def _get_projector(self, state, group):
        projector = group["projector"]
        if projector is None:
            return None
        if "projector" not in state:
            state["projector"] = copy.deepcopy(projector)
        return state["projector"]


class ProjectedMiniAdam(MiniAdam):
    def __init__(
        self,
        params,
        projector,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = True,
        update_gap: int = 200,
        experiment=None,
        model: torch.nn.Module | None = None,
        max_projected_params: int | None = None,
    ):
        param_groups = self._build_groups(
            params,
            projector,
            experiment,
            model,
            max_projected_params,
        )
        super().__init__(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            update_gap=update_gap,
        )

    def _build_groups(self, params, projector, experiment, model, max_projected_params):
        if model is not None:
            # Знаем имена — строим по одной группе на параметр
            groups = []
            projected_count = 0
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                can_project = param.dim() == 2 and projector is not None
                if max_projected_params is not None and projected_count >= max_projected_params:
                    can_project = False
                if can_project:
                    projected_count += 1
                groups.append({
                    "params": [param],
                    "projector": copy.deepcopy(projector) if can_project else None,
                    "param_name": build_param_name(name) if can_project else None,
                    "experiment": experiment,
                })
            return groups

        # Fallback: model не передан, имён нет
        if isinstance(params, dict):
            params = [params]

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            groups = []
            for group in params:
                group = dict(group)
                group.setdefault("projector", projector)
                group.setdefault("param_name", None)
                group.setdefault("experiment", experiment)
                groups.append(group)
            return groups

        return [{"params": list(params), "projector": projector, "param_name": None, "experiment": experiment}]


GaLoreMiniAdam = ProjectedMiniAdam