import copy

import torch


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
                    msg = "MiniAdam does not support sparse gradients"
                    raise RuntimeError(msg)

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
                        decay = 1.0 - group["lr"] * group["weight_decay"]
                        param.mul_(decay)
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
                    update = self._low_rank_update(
                        param, grad, state, group, beta1, projector
                    )
                else:
                    update = self._full_rank_update(param, grad, state, beta1)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = (exp_avg_sq / bias_correction2).sqrt()
                denom = denom.add_(group["eps"])
                update = update.div(bias_correction1)
                update = update.div(denom.to(update.dtype))

                param.add_(update, alpha=-group["lr"])

        return loss

    def _full_rank_update(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        beta1: float,
    ) -> torch.Tensor:
        if "exp_avg" not in state or state["exp_avg"].shape != param.shape:
            state["exp_avg"] = torch.zeros_like(param)

        exp_avg = state["exp_avg"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        return exp_avg

    def _low_rank_update(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        group: dict,
        beta1: float,
        projector,
    ) -> torch.Tensor:
        if (
            getattr(projector, "P", None) is None
            or state["step"] == 1
            or state["step"] % group["update_gap"] == 0
        ):
            projector.update_basis(grad)
            state.pop("exp_avg", None)

        low_rank_grad = projector.project(grad)
        has_exp_avg = "exp_avg" in state
        exp_avg_shape = state["exp_avg"].shape if has_exp_avg else None
        has_same_shape = exp_avg_shape == low_rank_grad.shape
        if not has_same_shape:
            state["exp_avg"] = torch.zeros_like(low_rank_grad)

        exp_avg = state["exp_avg"]
        exp_avg.mul_(beta1).add_(low_rank_grad, alpha=1.0 - beta1)
        return projector.reconstruct(exp_avg).to(param.dtype)

    def _get_projector(self, state: dict, group: dict):
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
    ):
        param_groups = self._with_projector(params, projector)
        super().__init__(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            update_gap=update_gap,
        )

    def _with_projector(self, params, projector):
        if isinstance(params, dict):
            params = [params]

        if isinstance(params, (list, tuple)):
            if len(params) > 0 and isinstance(params[0], dict):
                groups = []
                for group in params:
                    group = dict(group)
                    group.setdefault("projector", projector)
                    groups.append(group)
                return groups

            return [{"params": params, "projector": projector}]

        return [{"params": params, "projector": projector}]


GaLoreMiniAdam = ProjectedMiniAdam
