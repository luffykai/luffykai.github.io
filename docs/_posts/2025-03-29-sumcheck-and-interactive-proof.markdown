Notes from watching this [lecture](https://ocw.mit.edu/courses/6-5630-advanced-topics-in-cryptography-fall-2023/pages/lecture-1-interactive-proofs-and-the-sum-check-protocol/)

## Interactive Proof (IP)

todo

## Sumcheck protocol

Given a polynomial $f: \mathbb{F}^m \rightarrow \mathbb{F}$ of degree $d$,
i.e. there are $m$ variables, and each variable takes value in some field $\mathbb{F}$ and has degree at most $d$.
Fix some set $H \subset \mathbb{F}$, the **P** prover's goal is to convince **V** verifier the value of the sum of $f$ over $H$ if some value $\beta$:

$$\sum_{h_i \in H} f(h_1, h_2, ... h_m) = \beta$$


This is a sum over $|H|^m$ terms, so technically V can compute it but it's of exponential complexity.
The protocol works as follows:

**Round 1**

First, **V** asks **P** to send over an univariate polynomial

$$g_1(x) = \sum_{h2, h3, ..h_m \in H} f(x, h_2, ... h_m) $$

By assumption this is a univariate polynomial of degree at most $d$, so it can be represented as $d+1$ field elements.

Therefore, what **P** actually sends is: $\hat{g_1}(x) = \sum_{i=0}^d c_{1,i} x_i$ that claimed to be $g_1$.

If we sum over $g_1$ on $H$: $\sum_{x \in H} g_1(x) = $$\sum_{h_i \in H} f(h_1, h_2, ... h_m) = \beta$ (the last equal is "claimed to be", we will come back to that).
Therefore **V** should verify:
1. degree of $\hat{g}_1(x)$ at most $d$
1. $\sum_{h_1 \in H} \hat{g}_1(h_1) = \beta$.

If the above equality doesn't hold, **V** rejects as it's already clear that **P** is lying (either $\hat{g}_1$ is not $g_1$ or $\beta$ is not correct).
**V** then samples a random $t_1 \in \mathbb{F}$ as the challenge to **P**, with the idea that we want to "open" $f$ at $h_1 = t_1$.

**Round 2**

Next **V** asks **P** to send over the polynomial 

$$g_2(x) = \sum_{h_3, ... h_m} f(t_1, x, h_3, ..., h_m)$$

This "fixes" $h_1$ to be $t_1$ and thus reduce the number of variables by 1, recursion!
After **P** sends $\hat{g}_2$, **V** should verify
1. degree of $\hat{g}_2(x)$ at most $d$ 
1. $$\sum_{h_2 \in H}{\hat{g}_2(h_2)} = \sum_{h_2,h_3,...,h_m \in H} f(t_1, h_2, h_3, ..., h_m) = \hat{g}_1(t_1)$$

And **V** again samples a random $t_2 \in \mathbb{F}$ as the challenge to **P**.

**Round k**

At round $k$, **V** asks **P** to send over the polynomial 

$$g_k(x) = \sum_{h_{k+1}, ... h_m} f(t_1, ... t_{k-1}, x, h_{k+1}, ..., h_m)$$

After **P** sends $\hat{g}_k(x)$, **V** verifies that
1. degree of $\hat{g}_k(x)$ at most $d$ 
1. $$\sum_{h_k \in H}{\hat{g}_k(h_k)} = \hat{g}_{k-1}(t_{k-1})$$

And **V** again samples a random $t_k \in \mathbb{F}$ as the challenge to **P**.

**Final Round**

By repeating this process, **V** end up with a univariate polynomial $$\hat{g}_m(x)$$ that claimed to be equal to $g_m(x) = f(t_1, t_2, ... t_{m-1}, x)$.
**V** then samples a random $t_m \in \mathbb{F}$ and checks that
1. degree of $\hat{g}_m(x)$ at most $d$ 
1. $\hat{g}_m(t_m) = f(t_1, t_2, ... t_m)$

**Soundness analysis**

Why is this protocol sound? In a very informal way, we can think of the following, from the end back to the beginning:

$\hat{g}_m(x)$ has to be correct, i.e. $\hat{g}_m(x) = g_m(x) = f(t_1, t_2, ... x)$.
Otherwise it's very unlikely that $\hat{g}_m(x)$ and $g_m(x)$ agrees on a random selected $t_m$ (probability $d/|\mathbb{F}|$).

At any round $k$:

$$g_k(x) = \sum_h f(t_1, ... t_{k-1}, \color{red}{x}, \color{blue}{h_{k+1}}, h_{k+2}, ... h_m) $$

$$g_{k+1}(x) = \sum_h f(t_1, ... t_{k-1}, \color{red}{t_k}, \color{blue}{x}, h_{k+2}, ... h_m)$$

We have $g_k(t_k) = \sum_{x \in H} g_{k+1}(x)$.
Given that $\hat{g}_{k+1}(x)$ is correct, and the fact that **V** has verified (at round $k+1$): $$\sum_{h_{k+1} \in H}{\hat{g}_{k+1}(h_{k+1})} = \hat{g}_{k}(t_{k})$$,
$\hat{g}_k(t_k)$ is correct, and thus the $\hat{g}_k(x)$ has to be correct (with error probability $d/|\mathbb{F}|$)
because it agrees with $g_k(x)$ at a random point $t_k$.

Overall it's not hard to prove that the total soundness error is $md / \vert \mathbb{F} \vert$

**Final Notes**
- $f, g_1, g_2, ..$ are the true polynomials with known expressions.
- On the other hand $\hat{g}_i$ are the polynomial **P** sends that claimed to be equal to the above.
- The soundness error is independent of the size of $H$
- it's public coin protocol, so can be converted to non-interactive proof via Fiat-Shamir

