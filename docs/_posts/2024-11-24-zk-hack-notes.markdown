Writeups for the zk hack [puzzles](https://zkhack.dev/puzzles/).

## Puzzle-chaos-theory

In this [puzzle](https://github.com/ZK-Hack/puzzle-chaos-theory)
we break the authenticated encryption of ElGamal + BLS signature.

### Background

**Authenticated Encryption**

There are two security considerations when sending messages from A to B,
confidentiality (others without the secret key cannot decipher)
and authenticity (B gets the message intact, and is indeed from A).
An encryption algorithm turns the message (plaintext) into ciphertext so that

[Authenticated Encryption](https://en.wikipedia.org/wiki/Authenticated_encryption)
is an encryption scheme that achieve both properties, usually by combining an encryption algorithm
with a message authentication code [MAC](https://en.wikipedia.org/wiki/Message_authentication_code).

**ElGamal Encryption**

[ElGamal](https://en.wikipedia.org/wiki/ElGamal_encryption) is an asym encryption based on
[Diffie-Hellman](https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange).

Assuming A wants to send an encrypted message to B such that only B can decrypt.
As an asym encryption, B will first generate a pair of secret key (sk) and public key (pk) and publish the pk.
A will then encrypt the message $m$ with pk and send the ciphtertext $c$, and B can decrypt it with sk.

For ElGamal specifically, given a cyclic group $G$ with generator $g$ of order $q$. The plaintext message $m$ and ciphertext $c$ are $\in G$,
and private key (of receiver) $x = recv_sk \in [1, q)$. The public key is $g^{x} \in G$.
To encrypt $m$:

1. sender samples $y \in [1, q)$
2. with receiver's public key, compute $s = (g^x)^{y} = g^{xy}$. This is the "shared secret".
3. sender sends $(c_1, c_2) = (g^y, sm)$ to the receiver.

The receiver after receiving $(c_1, c_2)$ can first compute $s = c_1^x$, and thus $s^{-1}$ to recover $m$

**Pairing and BLS signature**

With [pairing](https://en.wikipedia.org/wiki/Pairing-based_cryptography) groups: $G_1, G_2, G_T$, the sender would to sign some message $h \in G_2$.
Sender samples a secret key $y \in [0, q)$. The public key is $g_1^y \in G_1$.
The signature is simply $sig = h^y \in G_2$.

To verify the signature, the receiver checks if the following equation holds:

$$e(g_1, sig) == e(g_1^y, h)$$

If everything is valid, both sides should be equal to $e(g_1, h)^y$

### Solution

In this puzzle, we are given the following data

```rust
pub struct Blob {
    pub sender_pk: G1Affine,
    pub c: ElGamal,
    pub sig: G2Affine,
    pub rec_pk: G1Affine,
}
```

In summary
- public information: $q, g1, g2$
- Sender secret: $m, y$
- Receiver secret: $x$
- shared secrets: $s = g1^{xy}$, only known by sender and receiver
- Sender sends ElGamal encryption: $c = (g_1^y, ms)$
- Sender sends BLS signature: $sig = h^y$, where $h \in G_2$ is the hash of above $c$ (thus $h$ is also known)

So what we know in both groups are
- $G1: g_1, g_1^x, g_1^y, mg_1^{xy}$
- $G2: g_2, h, h^y$

Notice that if we are given a message $m^{'}$, we can "decide" if it's the original message by
1. cancel out m to get s: $s^{'}= c_2 / m^{'}$
2. Check pairing $e(g_1^x, g^y) = e(s^{'}, h)$

Therefore we have the following solution

```rust
let target_output = Bls12_381::pairing(blob.rec_pk, blob.s);
let h = blob.c.hash_to_curve();
for (i, m) in messages.iter().enumerate() {
    if Bls12_381::pairing(blob.c.1 - m.0, h) == target_output {
        println!("matched!!! {}", i);
    }
}
```

Conclusion: Pairing breaks decisional Diffie-Hellman.

## Puzzle-supervillain

wip
