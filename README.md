# 🚀 cbrain

**The layer that understands consequences before an AI agent acts.**

---

## 🧠 What is cbrain?

Most AI systems today can:
- reason  
- plan  
- generate actions  

But they **don’t understand consequences**.

cbrain sits between planning and execution:

gbrain → plans action  
cbrain → evaluates risk & consequences  
PrivateVault → enforces decision  

---

## ⚠️ The Problem

AI agents are moving from:
- generating text → to executing real-world actions

But there’s a missing layer:

**Who decides if an action should even be attempted?**

Without it:
- destructive commands execute  
- financial actions go unchecked  
- irreversible operations slip through  

---

## ✅ The Solution

cbrain evaluates every action *before execution*:

- risk scoring  
- action classification  
- consequence modeling  
- decision recommendation (ALLOW / REVIEW / BLOCK)

---

## 🔥 Example

```bash
python run_flow.py transfer_funds
```

```
[gbrain] Planned action: transfer_funds

[cbrain] Action: transfer_funds
Risk Score: 0.9
Level: HIGH
Category: financial_irreversible
Decision: BLOCK

[PrivateVault] Enforcement:
RESULT: BLOCK
POLICY: pre-execution-risk-control
```

---

```bash
python run_flow.py read_system_logs
```

```
[gbrain] Planned action: read_system_logs

[cbrain] Action: read_system_logs
Risk Score: 0.2
Level: LOW
Category: read_only
Decision: ALLOW

[PrivateVault] Enforcement:
RESULT: ALLOW
POLICY: safe-observability
```

---

## 🧩 Architecture

[Agent / gbrain]  
        ↓  
[cbrain — consequence layer]  
        ↓  
[PrivateVault — execution control]  

---

## 💡 Key Ideas

- **Pre-execution intelligence**  
  Evaluate actions before they happen

- **Irreversibility awareness**  
  Detect destructive or financial operations

- **Decision-first systems**  
  Action is not the output — **decision is**

- **Separation of concerns**
  - gbrain → thinking  
  - cbrain → judgment  
  - PrivateVault → enforcement  

---

## ⚡ Why This Matters

The future of AI isn’t just smarter models.

It’s **controllable systems**.

The most important layer isn’t intelligence.  
It’s **decision + enforcement before execution**.

---

## 🛠 Quick Start

```bash
git clone https://github.com/LOLA0786/cbrain
cd cbrain

python run_flow.py transfer_funds
python run_flow.py read_system_logs
```

---

## 🔭 Roadmap

- dynamic authority modeling  
- multi-step action chain evaluation  
- rollback / irreversibility modeling  
- policy learning from past decisions  
- integration with real agent frameworks  

---

## 🧠 Positioning

gbrain remembers  
cbrain understands consequences  
PrivateVault enforces reality  

---

## ⚡ Final Thought

AI doesn’t fail at thinking.

It fails at **acting safely**.

cbrain exists to fix that.
