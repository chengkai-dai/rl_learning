# Policy Gradient

$\theta$ represents the **parameters** of your policy $\pi_{\theta}(a|s)$
$$
J(\theta)= \mathbb{E}_{\tau \sim \pi_{\theta}}(R(\tau))
$$
where $\tau$ is a trajectory represent a seqeunce of state-action.
$$
\tau=(s_0, a_0, s_1,a_1\dots,s_T,a_T)
$$
 and $R$ is the reward from $\tau$
$$
R(\tau)=\sum_{t=0}^{T}r(s_t,a_t)
$$
The probability of trajectory $\tau$ is 
$$
p_{\theta}(\tau)=p(s_0)\prod_{t=0}^{T}\pi_{\theta}(a_r|s_t)p(s_{t+1}|s_t,a_t)
$$
Specifically,
$$
p_{\theta}(\tau)=p(s_0)\cdot\pi_{\theta}(a_0|s_0)\cdot p(s_1|s_0,a_0)\cdot \pi_{\theta}(a_1|a_0)\cdot p(s_2|s_1,a_1)\dots\pi_{\theta}(a_T|s_{T})\cdot p(s_{T+1}|s_T,a_T)
$$
The goal is to get the gradient of $J_{\theta}$, which is $\nabla_{\theta} J_{\theta}$,
$$
\nabla_{\theta} J(\theta)=\nabla_{\theta}\mathbb{E}_{\tau \sim \pi_{\theta}}(R(\tau))
$$
First we formulate the problem as intergral form
$$
\nabla_{\theta}J(\theta)=\nabla_{\theta}\int_{\tau}p_{\theta}(\tau)R(\tau)d\tau
$$
Equiventally,
$$
\nabla_{\theta}J(\theta)=\int_{\tau}\nabla_{\theta}[p_{\theta}(\tau)]R(\tau)d\tau
$$


using the log-gradient trick,
$$
\nabla_{\theta}p_{\theta}(x)=p_{\theta}(x)\nabla_{\theta}\log p_{\theta}(x)
$$


we get,
$$
\nabla_{\theta}J(\theta)=\int_{\tau}p_\theta(\tau)\nabla_{\theta}\log p_\theta(\tau) R(\tau)d\tau
$$
change to the sum form,
$$
\nabla_{\theta}J(\theta)=\sum_{\tau}p_\theta(\tau)\nabla_{\theta}\log p_\theta(\tau) R(\tau)d\tau
$$
we get the expection form,
$$
\nabla_{\theta}J(\theta)=\mathbb{E}_{\tau \sim \pi_{\theta}}[\nabla_{\theta}\log p_\theta(\tau) R(\tau)]
$$
From the definition of $p_{\theta}(\tau)$, we get
$$
\log p_{\theta}(\tau)=\log p(s_0)+ \sum_{t=0}^T \log \pi_{\theta}(a_t|s_t)+\sum_{t=0}^T \log p(s_{t+1}|s_t,a_t)
$$




