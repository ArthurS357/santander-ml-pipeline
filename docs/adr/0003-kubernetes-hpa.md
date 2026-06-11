# ADR 0003 — Kubernetes + HPA para escalabilidade

## Status

Aceito.

## Contexto

O pilar de Escalabilidade exige dimensionamento horizontal. É preciso provar
que a API escala sob carga e se recupera de falhas, com probes de saúde
corretas.

## Decisão

Empacotar a API em container e implantar em **Kubernetes** com **HPA**
(Horizontal Pod Autoscaler).

- `Deployment` com 3 réplicas, `RollingUpdate` (`maxUnavailable: 0`).
- **Probes separadas:** `livenessProbe` → `/health/live` (processo vivo);
  `readinessProbe` → `/health/ready` (503 sem modelo, bloqueia tráfego).
- `HPA` 3–10 réplicas (CPU 70% / memória 80%).
- Hardening: `runAsNonRoot`, `readOnlyRootFilesystem`, `drop ALL` capabilities.
- Imagem `:latest` na PoC; `:${github.sha}` (imutável) em produção (ADR implícito
  na esteira de CI).

## Consequências

- **Positivas:** autoscaling real, zero-downtime deploy, isolamento de falha por
  probe (não roteia tráfego para pod sem modelo).
- **Negativas / trade-offs:** exige cluster K8s; segredos devem vir de
  Secret/Vault (o manifesto versiona apenas placeholders).
