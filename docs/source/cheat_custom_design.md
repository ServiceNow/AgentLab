# Cheat Custom Design (Action-Space Trajectories)

This document describes a low-impact refactor to add a new `cheat_custom` routine that emits **BrowserGym high‑level action strings** (agent‑format actions), while keeping the existing `cheat()` intact. The goal is to obtain trajectories whose `step.action` matches the agent action space (e.g., `click("bid")`, `fill("bid","text")`, `send_msg_to_user("...")`).

The design is split into two parts:

- **Part 1: General codebase changes** (AgentLab + small glue logic)
- **Part 2: Per‑task implementation** (AgentLab-side adapters; WorkArena code stays untouched)

> IMPORTANT: This document is a **design and workflow guide** only. It does not modify code.

---

## Overview & Goals

**Goal**: Produce “oracle” trajectories whose **actions are in the same format as agent predictions** (BrowserGym high‑level action strings), without breaking existing `cheat()` behavior.

**Confirmed action space** (this is the exact format the agent predicts and we must emit):
- `noop(wait_ms: float = 1000)`
- `scroll(delta_x: float, delta_y: float)`
- `fill(bid: str, value: str, enable_autocomplete_menu: bool = False)`
- `select_option(bid: str, options: str | list[str])`
- `click(bid: str, button: Literal['left','middle','right'] = 'left', modifiers: list[Literal['Alt','Control','ControlOrMeta','Meta','Shift']] = [])`
- `dblclick(bid: str, button: Literal['left','middle','right'] = 'left', modifiers: list[Literal['Alt','Control','ControlOrMeta','Meta','Shift']] = [])`
- `hover(bid: str)`
- `press(bid: str, key_comb: str)`
- `focus(bid: str)`
- `clear(bid: str)`
- `drag_and_drop(from_bid: str, to_bid: str)`
- `tab_focus(index: int)`
- `new_tab()`
- `tab_close()`
- `go_back()`
- `go_forward()`
- `goto(url: str)`
- `send_msg_to_user(text: str)`
- `report_infeasible(reason: str)`

**Key constraints**:
- WorkArena’s existing `cheat()` executes Playwright directly and returns `None`.
- To get agent‑style actions, we need a new routine that returns action strings **instead of executing**.
- We should avoid impacting existing logic or evaluation behavior unless explicitly using `cheat_custom`.
 - Changes are limited to `enterprise/AgentLab/` only.

**Non‑goals**:
- Rewriting the entire environment or action space.
- Changing benchmark definitions unless required for new tasks.
 - Modifying WorkArena task code.

**Compatibility requirement**:
- Keep trajectory/log layout compatible with existing imports (e.g., `enterprise/data/trajectories/20260121_import/...`):
  - same per-episode folder structure
  - same `step_*.pkl.gz` format (StepInfo serialization)
  - `step.action` is a string in agent action format
  - any new metadata goes into `agent_info` to preserve backward compatibility

**Additional knowledge gathered**:
- WorkArena-Reference does **not** define the action space; it is embedded in the observation prompt.
- WorkArena `cheat()` uses Playwright and returns `None` (per-task actions are not directly available).
- WorkArena-Reference includes a `wa_action_traces.py` monkey‑patch example (useful reference, not required for this design).

---

# Part 1 — General Codebase Changes (AgentLab + Glue)

This section describes changes needed once, in the AgentLab codebase and the cheating agent logic. These changes are designed to be minimal and backward‑compatible.

## Part 1 Status (Completed)

Part 1 has been implemented in `enterprise/AgentLab/` with the following concrete changes:

- **Registry + stubs added**: `src/agentlab/cheat_custom/registry.py` and `src/agentlab/cheat_custom/__init__.py`.
  - Provides a registry for `cheat_custom` implementations.
  - Populates **stubs for all WorkArena tasks** via `browsergym.workarena.ALL_WORKARENA_TASKS`.
  - Attaches `cheat_custom` to task classes/instances on demand.
- **CheatingAgent defaults to cheat_custom**: `src/agentlab/agents/cheating_agent.py`.
  - `cheat_method` default is now `cheat_custom`.
  - `cheat_custom` path **fails fast** on missing or empty actions.
  - `ensure_cheat_custom(task)` is called so each task has a `cheat_custom` method (real or stub).
  - **Compositional subtasks now advance via `valid_index`**, allowing multiple `cheat_custom` calls per subtask (multi‑phase flows).
- **Backwards compatibility preserved**:
  - Existing `cheat()` behavior remains unchanged when `cheat_method="cheat"`.
  - Trajectory/log layout is unchanged; `step.action` is now the real action string when `cheat_custom` is used.

## 1.1 New Contract: `cheat_custom` API

**Add a new optional method** on task classes:

```
cheat_custom(page, chat_messages, subtask_idx=None) -> list[str]
```

- **Returns** a list of BrowserGym high‑level action strings **in the exact action space above**.
- **Must not** perform browser actions itself (no direct clicks/fills), except read‑only DOM access if needed.
- If the task is compositional, it must accept `subtask_idx` and return only actions for that subtask.

**Backwards compatibility**:
- Existing `cheat()` stays unchanged.
- `cheat_custom` is only used when explicitly called.

**Required error behavior**:
- When `cheat_custom` is selected and a task does not provide it (or returns empty/invalid), raise a clear error. No silent fallback.

## 1.2 New Agent Behavior (CheatingAgent)

Update the CheatingAgent to prefer `cheat_custom` if present:

**High‑level algorithm**:
1. On each step, check if task has `cheat_custom`.
2. If yes:
   - If no pending actions in the queue, call `cheat_custom` to get a list of actions.
   - Return actions **one by one** on each step.
   - When list is exhausted, move to next subtask (if compositional) or finish.
3. If no `cheat_custom`, and `cheat_custom` mode is enabled, **raise an error** (fail fast).
4. If `cheat_custom` mode is disabled, fall back to the old `cheat()` behavior (Playwright‑based).

**Why this preserves current logic**:
- Default behavior remains unchanged.
- Only the cheating agent uses the new routine, and only if it exists.

## 1.3 Action Queue Semantics

To ensure `step.action` is meaningful:

- Each call to `cheat_custom` returns a list of action strings.
- The agent uses a queue to emit them across steps, one per step.
- Do **not** inject fallback actions unless the queue is empty and the task requires a no‑op to trigger validation. Prefer **real actions only**.

## 1.4 Compositional Tasks (L2/L3)

Compositional tasks have subtasks that must be solved in order:

- Add a **subtask index** tracker in the agent.
- For each subtask:
  - call `cheat_custom(..., subtask_idx)` to get the action list for that subtask.
  - execute the list in order.
  - once actions are done, advance to next subtask.

This is necessary because WorkArena validation skips unvalidated subtasks; using `valid_index` will skip steps.

## 1.5 Logging & Output Expectations

- `step.action` should now be the actual action string.
- Use `send_msg_to_user("...")` to represent answers previously injected directly into chat.
- If a task only needs to post chat, the action list can be a single `send_msg_to_user(...)`.

## 1.6 Testing Strategy (General)

Create a small test harness that:
- Runs 1–2 tasks per level.
- Confirms:
  - `cheat_custom` exists and returns list[str].
  - `step.action` matches expected formats.
  - Environment validates success.

## 1.7 AgentLab‑Only Implementation Strategy

We will not modify WorkArena task code. Instead, implement `cheat_custom` via **AgentLab-side adapters**:

- Add an AgentLab registry that maps task classes to `cheat_custom` implementations.
- At runtime, attach a bound `cheat_custom` method to the task instance (or wrap the task).
- Provide **stub implementations for every task** that raise `NotImplementedError` with a clear message.
- For tasks that have been implemented, the registry overrides the stub.

This ensures:
- You can call `task.cheat_custom()` directly (as requested).
- WorkArena code remains untouched.
- Missing tasks fail fast.

---

# Part 2 — Per‑Task Implementation Workflow (AgentLab Adapters)

This is the heavy‑lift section. Each task that needs action‑space trajectories must implement `cheat_custom` **as an AgentLab adapter**, not by editing WorkArena.

## Part 2 Status (In Progress)

**Implemented adapters (L1 pilot):**
- `workarena.servicenow.all-menu` → `goto(final_url)`
- `workarena.servicenow.filter-incident-list` → `goto(start_url?sysparm_query=...)`
- `workarena.servicenow.create-incident` → form actions (`fill`, `select_option`, `click`)
- `workarena.servicenow.order-apple-watch` → multi‑phase catalog flow (Hardware → item → quantity → order)

**Implemented adapters (L3 pilot):**
- `workarena.servicenow.navigate-and-create-incident-l3` → compositional delegation
- `workarena.servicenow.navigate-and-filter-incident-list-l3` → compositional delegation
- `workarena.servicenow.navigate-and-order-apple-watch-l3` → compositional delegation

**Building‑block adapter (used by L3):**
- `UpdatePrivateTask` → list search + open record → set state + update (multi‑phase)

**Implemented adapters (L2 Batch 1):**
- `InfeasibleCompositionalTask` → `report_infeasible(...)` on infeasible subtask, delegate otherwise
- `FilterListTask` → `goto(start_url?sysparm_query=...)` (supports `contains`, `equals`, etc.)
- `SendChatMessageForBudgetAllocationTask` → `send_msg_to_user(...)`
- `DeleteRecordTask` (+ expense line subclasses) → delete via action-space adapter (fallback to API delete when needed)
- Batch 1 task IDs: all infeasible navigate‑and‑do L2, expense management L2, maximize investment return L2 (see §2.7.3)

**Coverage so far:** 4 / 33 L1 tasks, 3 pilot L3 tasks, 65 L2 bases (67 IDs) in Batch 1.

**Prioritization guidance:** Use `enterprise/AgentLab/longmemevalv2_trajectory_collection_journal.json` to prioritize which tasks to implement next (focus on tasks with no successful trajectory).

**Known gaps / risks discovered:**
- `Order*` tasks can be **multi‑phase**; `cheat_custom` must be callable more than once for a single task.
- Form tasks require **tab switching** to surface fields; `cheat_custom` needs to click tab headers before filling.
- List filter tasks can be simplified by **direct URL query** when reference/choice values are mapped to internal values.

## 2.1 General Per‑Task Workflow

For each task class:

1. **Study the existing `cheat()`**
   - Identify every UI action it performs: click, fill, select, press, navigation, etc.
   - Identify any chat message additions (`chat_messages.append(...)`).

2. **Map each Playwright action to BrowserGym action string** (use the confirmed action space list)
   - `click(bid)` for clicks on elements with `bid`.
   - `fill(bid, value)` for text entry.
   - `select_option(bid, value)` for dropdowns.
   - `press(bid, key)` for key events on a particular element.
   - `send_msg_to_user(text)` for chat outputs.

3. **Resolve `bid` for each element**
   - Use Playwright **read‑only** operations to locate the element and read `bid`:
     - `element_handle.evaluate('(el) => el.getAttribute("bid")')`
   - If the element doesn’t have a `bid`:
     - Prefer locating a **nearby element** that does (e.g., the input field instead of its container).
     - If still missing, note as a task that requires deeper UI annotation changes.

4. **Compose the action list** in the order the agent would execute.

5. **Return action list** (do not execute Playwright actions).

6. **Ensure action subset compatibility**
   - WorkArena action subset includes: `click, fill, select_option, press, focus, clear, drag_and_drop, scroll, send_msg_to_user`.
   - Avoid actions outside this list unless you also update the action set (not recommended for now).

7. **For compositional tasks**
   - Implement `cheat_custom(..., subtask_idx)` by delegating to the subtask’s `cheat_custom`.
   - For parent tasks, assemble per‑subtask sequences.

## 2.2 Handling Chat‑Only Tasks

Some tasks use `cheat()` purely to insert a correct answer into chat.

- In `cheat_custom`, **return a single action**:
  - `send_msg_to_user("<answer>")`
- The answer can be derived via read‑only DOM queries or direct task metadata.

## 2.3 Handling Navigation / List Tasks

For tasks like `FilterListTask` or navigation:

- Convert filter setup steps into:
  - `click(...)` for filter UI
  - `select_option(...)` for fields/operators
  - `fill(...)` for values
  - `click(...)` for “Run”

Ensure each UI control has a valid `bid`. If not, you may need to identify the actual input element (which usually has a `bid`).

## 2.4 Specific High‑Risk Areas

These will be the most difficult to convert and should be prioritized:

- **Dashboards / Charts** (requires locating report/visual elements with `bid`).
- **Compositional tasks** with nested subtasks that include navigation + filtering + record edits.
- Tasks that rely on **dynamic pop‑ups** or frame‑specific locators.

## 2.5 Suggested Task Order

1. L1 atomic tasks (simpler, smaller surface area)
2. L2 tasks with shallow subtasks
3. L3 compositional tasks

## 2.7 L2 Batch 1 Design (Infeasible + Expense/Investment)

**Batch scope (from `longmemevalv2_trajectory_collection_journal.json`):** 65 base tasks / 67 missing L2 task IDs.

**Modules covered:**
- `browsergym.workarena.tasks.compositional.navigate_and_do_infeasible` (27 bases / 29 IDs)
- `browsergym.workarena.tasks.compositional.expense_management` (11 bases / 11 IDs)
- `browsergym.workarena.tasks.compositional.maximize_investment_return` (27 bases / 27 IDs)

### 2.7.1 Common mechanics we will implement once

1) **Infeasible reporting**
   - Tasks subclass `InfeasibleCompositionalTask`; validation expects the last chat message to have role `"infeasible"`.
   - `cheat_custom` should return a single `report_infeasible(reason)` action **on the infeasible subtask**.
   - If `task.provide_reason` is `True`, use `", ".join(task.infeasible_reasons)` (or the first reason).
   - If `task.provide_reason` is `False`, pass an empty string (the infeasible reasons list is `[""]`, so empty is acceptable).
   - For non‑infeasible subtasks in the same compositional chain, emit a minimal action (`noop()` or `goto(...)`) to keep the action queue non‑empty.

2) **Generic list filtering (FilterListTask)**
   - Implement `cheat_custom` for `FilterListTask` to **skip UI clicks** and instead `goto()` the list URL with a `sysparm_query` built from:
     - `filter_columns`, `filter_operators`, `filter_values`, and `filter_kind` (AND/OR).
   - Reuse the same query‑builder logic across Expense/Investment tasks.

3) **Generic record deletion (DeleteRecordTask)**
   - Implement `cheat_custom` for `DeleteRecordTask` (and its subclasses) to:
     - `goto(list_url?sysparm_query=<field>=<value>)`
     - open the record (by number or first row)
     - click **Delete** and confirm.
   - This is shared by `DeleteExpenseLineExpenseManagementTask` and `DeleteExpenseLineKnapsack`.

4) **Chat message emission**
   - Implement `cheat_custom` for `SendChatMessageForBudgetAllocationTask` to return
     `send_msg_to_user(self.message)`.
   - This covers all “total return”, “investments only”, and “return + investments” variants.

### 2.7.2 Batch‑specific design notes

**A) Infeasible Navigate‑and‑Do tasks**
- These L2 tasks use two subtasks: `AllMenuTask` (not validated) + an infeasible task (also not validated).
- Validation only checks for the infeasible chat message, so the adapter should:
  - return a small action list for the first subtask (e.g., `noop()` or `goto(task.start_url)`), and
  - return `report_infeasible(...)` for the last subtask.
- No actual form/list/catalog action is required.

**B) Expense Management tasks**
- These are `FilterAndDoTask` subclasses that create duplicate expense lines and require deletion of specific rows.
- Subtasks include: `AllMenuTask`, `FilterListTask`, and multiple `DeleteExpenseLineExpenseManagementTask` entries.
- Adapter strategy:
  - For `AllMenuTask`, reuse the existing `all-menu` adapter (`goto(final_url)`).
  - For `FilterListTask`, use the generic sysparm query `goto`.
  - For each delete subtask, use the generic `DeleteRecordTask` adapter.

**C) Maximize Investment Return tasks**
- These tasks reuse the same list/filter mechanics and add **chat output** and/or **deletions**.
- Subtasks include `AllMenuTask`, `FilterListTask`, `SendChatMessageForBudgetAllocationTask`, and possibly `DeleteExpenseLineKnapsack`.
- Adapter strategy:
  - `SendChatMessageForBudgetAllocationTask` → `send_msg_to_user(self.message)`
  - `DeleteExpenseLineKnapsack` → generic delete adapter
  - Filtering/navigation as above.

### 2.7.3 Batch 1 task IDs (bases)

**navigate_and_do_infeasible**
- workarena.servicenow.infeasible-navigate-and-create-change-request-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-create-hardware-asset-l2
- workarena.servicenow.infeasible-navigate-and-create-hardware-asset-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-create-incident-l2
- workarena.servicenow.infeasible-navigate-and-create-problem-l2
- workarena.servicenow.infeasible-navigate-and-create-user-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-filter-asset-list-l2
- workarena.servicenow.infeasible-navigate-and-filter-asset-list-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-filter-change-request-list-l2
- workarena.servicenow.infeasible-navigate-and-filter-change-request-list-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-filter-hardware-list-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-filter-incident-list-l2
- workarena.servicenow.infeasible-navigate-and-filter-user-list-l2
- workarena.servicenow.infeasible-navigate-and-order-apple-watch-l2
- workarena.servicenow.infeasible-navigate-and-order-developer-laptop-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-order-ipad-mini-l2
- workarena.servicenow.infeasible-navigate-and-order-ipad-mini-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-order-ipad-pro-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-order-loaner-laptop-l2
- workarena.servicenow.infeasible-navigate-and-order-standard-laptop-l2
- workarena.servicenow.infeasible-navigate-and-sort-asset-list-l2
- workarena.servicenow.infeasible-navigate-and-sort-asset-list-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-sort-hardware-list-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-sort-incident-list-l2
- workarena.servicenow.infeasible-navigate-and-sort-incident-list-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-sort-service-catalog-item-list-with-reason-l2
- workarena.servicenow.infeasible-navigate-and-sort-user-list-l2

**expense_management**
- workarena.servicenow.amount-based-expense-management-large-l2
- workarena.servicenow.amount-based-expense-management-medium-l2
- workarena.servicenow.basic-expense-management-large-l2
- workarena.servicenow.basic-expense-management-medium-l2
- workarena.servicenow.basic-expense-management-small-l2
- workarena.servicenow.date-based-expense-management-large-l2
- workarena.servicenow.date-based-expense-management-medium-l2
- workarena.servicenow.date-based-expense-management-small-l2
- workarena.servicenow.easy-expense-management-large-l2
- workarena.servicenow.easy-expense-management-medium-l2
- workarena.servicenow.easy-expense-management-small-l2

**maximize_investment_return**
- workarena.servicenow.filter-random-expenses-and-delete-wrong-investments-medium-l2
- workarena.servicenow.filter-random-expenses-and-find-total-return-large-l2
- workarena.servicenow.filter-random-expenses-and-find-total-return-medium-l2
- workarena.servicenow.filter-random-expenses-and-find-total-return-small-l2
- workarena.servicenow.filter-random-expenses-and-select-investments-large-l2
- workarena.servicenow.filter-random-expenses-and-select-investments-medium-l2
- workarena.servicenow.filter-random-expenses-and-select-investments-small-l2
- workarena.servicenow.filter-random-expenses-find-total-return-and-select-investments-medium-l2
- workarena.servicenow.filter-single-item-expenses-and-delete-wrong-investments-large-l2
- workarena.servicenow.filter-single-item-expenses-and-delete-wrong-investments-medium-l2
- workarena.servicenow.filter-single-item-expenses-and-find-total-return-large-l2
- workarena.servicenow.filter-single-item-expenses-and-find-total-return-medium-l2
- workarena.servicenow.filter-single-item-expenses-and-find-total-return-small-l2
- workarena.servicenow.filter-single-item-expenses-and-select-investments-medium-l2
- workarena.servicenow.filter-single-item-expenses-find-total-return-and-select-investments-medium-l2
- workarena.servicenow.filter-single-item-uniform-expenses-and-delete-wrong-investments-small-l2
- workarena.servicenow.filter-single-item-uniform-expenses-and-select-investments-large-l2
- workarena.servicenow.filter-single-item-uniform-expenses-and-select-investments-medium-l2
- workarena.servicenow.filter-single-item-uniform-expenses-find-total-return-and-select-investments-medium-l2
- workarena.servicenow.filter-three-items-uniform-expenses-and-select-investments-medium-l2
- workarena.servicenow.filter-three-items-uniform-expenses-find-total-return-and-select-investments-large-l2
- workarena.servicenow.filter-three-items-uniform-expenses-find-total-return-and-select-investments-medium-l2
- workarena.servicenow.filter-trivial-expenses-and-find-total-return-large-l2
- workarena.servicenow.filter-trivial-expenses-and-select-investments-large-l2
- workarena.servicenow.filter-trivial-expenses-find-total-return-and-select-investments-large-l2
- workarena.servicenow.filter-trivial-expenses-find-total-return-and-select-investments-small-l2
- workarena.servicenow.filter-two-items-uniform-expenses-and-select-investments-small-l2

## 2.6 Validation Checklist per Task

For each task with `cheat_custom`:

- [ ] `cheat_custom` returns list[str]
- [ ] All actions are valid in WorkArena action subset
- [ ] Every action has a valid `bid` or acceptable parameters
- [ ] Task completes and validates successfully
- [ ] `step.action` reflects real actions

---

# Risks & Mitigations

**Risk: Missing BIDs**
- Mitigation: Identify alternative clickable elements with BIDs.
- If unavoidable, consider UI annotation changes (outside current scope).

**Risk: Non‑deterministic UI elements**
- Mitigation: Use deterministic selectors to find BIDs in `cheat_custom` (read‑only).

**Risk: Task behavior depends on Playwright timing**
- Mitigation: Since `cheat_custom` is not executing actions, the timing is now controlled by the environment; you may need to add small `scroll` or `wait` actions only if absolutely needed.

---

# Deliverables

**Part 1 (General Codebase)**
- Add `cheat_custom` contract in tasks (optional method).
- Update CheatingAgent to prefer `cheat_custom` and execute returned action strings sequentially.
- Ensure compositional tasks iterate subtasks in order.
- Implement AgentLab registry/adapters and stub methods per task (fail fast).
- Keep trajectory/log formats compatible with existing runs.

**Part 2 (Per‑Task)**
- Implement `cheat_custom` in AgentLab adapters for each WorkArena task.
- Map Playwright actions to BrowserGym action strings using element BIDs.
- Validate each task’s action trajectory and ensure agent‑format action strings are produced.

---

# Suggested Handoff Notes

- `enterprise/WorkArena-Reference` is the authoritative source for task logic, but **do not modify it**.
- Focus on creating **action‑string trajectories**, not on changing behavior of `cheat()`.
- Start with a small L1 subset to validate the end‑to‑end pipeline.

## Next Planned Work (Part 2)

1) Expand L1 coverage by category:
   - **List filters:** filter‑asset, filter‑change‑request, filter‑hardware, filter‑service‑catalog‑item, filter‑user
   - **Forms:** create‑change‑request, create‑hardware‑asset, create‑problem, create‑user
   - **Catalog:** order‑apple‑mac‑book‑pro15, order‑developer‑laptop, order‑development‑laptop‑p‑c, order‑ipad‑mini, order‑ipad‑pro, order‑loaner‑laptop, order‑sales‑laptop, order‑standard‑laptop
   - **Dashboards:** single‑chart value/min‑max, multi‑chart value/min‑max
2) After L1 completion, move to L2/L3 compositional subtasks in priority order.
