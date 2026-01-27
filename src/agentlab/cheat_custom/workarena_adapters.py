import logging
from urllib import parse

from agentlab.cheat_custom.registry import register_cheat_custom, ensure_cheat_custom

logger = logging.getLogger(__name__)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _bid_from_locator(locator, name: str) -> str:
    if locator is None:
        raise RuntimeError(f"Missing locator for {name}")
    if locator.count() == 0:
        raise RuntimeError(f"No elements found for {name}")
    handle = locator.first.element_handle()
    if handle is None:
        raise RuntimeError(f"No element handle for {name}")
    bid = handle.evaluate("(el) => el.getAttribute('bid')")
    if not bid:
        raise RuntimeError(f"No bid attribute for {name}")
    return bid


def _bid_from_handle(handle, name: str) -> str:
    if handle is None:
        raise RuntimeError(f"Missing handle for {name}")
    element = handle.as_element() if hasattr(handle, "as_element") else handle
    if element is None:
        raise RuntimeError(f"Handle is not an element for {name}")
    bid = element.evaluate("(el) => el.getAttribute('bid')")
    if not bid:
        raise RuntimeError(f"No bid attribute for {name}")
    return bid


def _append_query(url: str, query: str) -> str:
    if not query:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}sysparm_query={parse.quote(query, safe='')}"


def _build_filter_query(task) -> str:
    from browsergym.workarena.api.utils import table_api_call

    columns = task.filter_columns
    values = task.filter_values
    operators = getattr(task, "filter_operators", None) or ["is"] * len(columns)
    kind = task.filter_kind
    list_info = task.list_info

    predicates = []
    for col, val, op in zip(columns, values, operators):
        op_norm = str(op or "is").strip().lower()

        if op_norm in {"is empty", "isempty"} or val is None or val == "":
            predicates.append(f"{col}ISEMPTY")
            continue
        if op_norm in {"is not empty", "isnotempty"}:
            predicates.append(f"{col}ISNOTEMPTY")
            continue

        col_info = list_info["columns"][col]
        col_type = col_info.get("type")
        mapped_val = val

        if col_type == "choice":
            choices = col_info.get("choices", {})
            for key, display in choices.items():
                if display == val:
                    mapped_val = key
                    break

        if col_type == "reference" and op_norm in {"is", "=", "equals", "equal"}:
            ref_table = col_info.get("reference")
            ref_field = col_info.get("reference_attributes", {}).get("display_field")
            if ref_table and ref_field:
                try:
                    res = table_api_call(
                        instance=task.instance,
                        table=ref_table,
                        params={
                            "sysparm_query": f"{ref_field}={val}",
                            "sysparm_fields": "sys_id",
                            "sysparm_limit": "1",
                        },
                    )["result"]
                except Exception as exc:
                    raise RuntimeError(f"Failed to resolve reference {col}={val}: {exc}")
                if res:
                    mapped_val = res[0]["sys_id"]

        if op_norm in {"is", "=", "equals", "equal"}:
            predicates.append(f"{col}={mapped_val}")
        elif op_norm in {"is not", "!=", "not", "is not equal to"}:
            predicates.append(f"{col}!={mapped_val}")
        elif op_norm in {"contains", "contain"}:
            predicates.append(f"{col}LIKE{mapped_val}")
        elif op_norm in {"does not contain", "not contains", "not contain"}:
            predicates.append(f"{col}NOT LIKE{mapped_val}")
        elif op_norm in {"starts with", "startswith"}:
            predicates.append(f"{col}STARTSWITH{mapped_val}")
        elif op_norm in {"ends with", "endswith"}:
            predicates.append(f"{col}ENDSWITH{mapped_val}")
        elif op_norm in {"greater than", ">"}:
            predicates.append(f"{col}>{mapped_val}")
        elif op_norm in {"less than", "<"}:
            predicates.append(f"{col}<{mapped_val}")
        else:
            predicates.append(f"{col}={mapped_val}")

    sep = "^OR" if kind == "OR" else "^"
    return sep.join(predicates)


def cheat_custom_all_menu(task, page=None, chat_messages=None, subtask_idx=None):
    url = getattr(task, "final_url", None) or getattr(task, "start_url", None)
    if not url:
        raise RuntimeError("AllMenuTask missing final_url/start_url")
    return [f"goto('{_escape(url)}')"]


def cheat_custom_filter_incident_list(task, page=None, chat_messages=None, subtask_idx=None):
    query = _build_filter_query(task)
    url = _append_query(task.start_url, query)
    return [f"goto('{_escape(url)}')"]


def cheat_custom_filter_list(task, page=None, chat_messages=None, subtask_idx=None):
    query = _build_filter_query(task)
    url = _append_query(task.start_url, query)
    return [f"goto('{_escape(url)}')"]


def _get_form_tab_bid(task, page, field: str) -> str | None:
    section = page.evaluate(
        f"""() => {{
        const element = {task.form_js_selector}.getElement('{field}');
        if (!element) return null;
        const ancestors = element.ancestors();
        for (let ancestor of ancestors) {{
            if (ancestor.id && ancestor.id.startsWith('section-')) {{
                return ancestor.id;
            }}
        }}
        return null;
    }}"""
    )
    if not section:
        return None
    section_id = section.split("-")[-1]
    tab_ids = page.evaluate(f"{task.js_prefix}.g_tabs2Sections.tabIDs")
    tab_sections = {s.split(".")[-1]: i for i, s in enumerate(tab_ids)}
    if section_id not in tab_sections:
        return None
    idx = tab_sections[section_id]
    handle = page.evaluate_handle(f"{task.js_prefix}.g_tabs2Sections.tabsTabs[{idx}].element")
    return _bid_from_handle(handle, f"tab for field {field}")


def cheat_custom_create_incident(task, page, chat_messages=None, subtask_idx=None):
    task._wait_for_ready(page, iframe_only=True)
    iframe = page.frame(task.js_prefix)
    if iframe is None:
        raise RuntimeError("Could not find gsft_main iframe for form")

    new_btn = iframe.locator("#sysverb_new")
    if new_btn.count() > 0:
        return [f"click('{_bid_from_locator(new_btn, 'New button')}')"]

    if task.table_metadata is None:
        task._get_form(page)
    if task.fields is None:
        task._get_fields(page)

    actions: list[str] = []
    clicked_tabs: set[str] = set()

    for field in task.task_fields:
        tab_bid = _get_form_tab_bid(task, page, field)
        if tab_bid and tab_bid not in clicked_tabs:
            actions.append(f"click('{tab_bid}')")
            clicked_tabs.add(tab_bid)

        label = page.evaluate(f"{task.form_js_selector}.getLabelOf('{field}')")
        control = iframe.get_by_label(label, exact=True)
        if control.count() > 1:
            control = control.nth(0)
        bid = _bid_from_locator(control, f"field {field}")
        value = task.template_record[field]
        field_type = task.table_metadata[field]["type"]

        if field_type == "choice":
            if value is None or value == "":
                continue
            actions.append(f"select_option('{bid}', '{_escape(str(value))}')")
        elif field_type == "boolean":
            desired = str(value).lower() == "true"
            try:
                current = control.is_checked()
            except Exception:
                current = None
            if current is None or current != desired:
                actions.append(f"click('{bid}')")
        else:
            if value is None or value == "":
                continue
            enable_ac = control.get_attribute("aria-autocomplete") == "list"
            if enable_ac:
                actions.append(
                    f"fill('{bid}', '{_escape(str(value))}', True)"
                )
            else:
                actions.append(f"fill('{bid}', '{_escape(str(value))}')")

    submit_btn = iframe.locator("#sysverb_insert")
    actions.append(f"click('{_bid_from_locator(submit_btn, 'Submit button')}')")

    return actions


def cheat_custom_order_apple_watch(task, page, chat_messages=None, subtask_idx=None):
    task._wait_for_ready(page)
    iframe = page.frame(task.js_prefix)
    if iframe is None:
        raise RuntimeError("Could not find gsft_main iframe for catalog")

    url = page.evaluate("() => window.location.href")
    if "servicecatalog_cat_item_view" in url:
        actions: list[str] = []
        quantity_select = iframe.locator("#quantity")
        actions.append(
            f"select_option('{_bid_from_locator(quantity_select, 'quantity')}', '{task.quantity}')"
        )
        order_btn = iframe.locator("#oi_order_now_button")
        actions.append(f"click('{_bid_from_locator(order_btn, 'order now')}')")
        return actions

    item_heading = iframe.locator(f"h2:has-text('{task.requested_item}')")
    if item_heading.count() > 0 and item_heading.first.is_visible():
        return [f"click('{_bid_from_locator(item_heading, 'item heading')}')"]

    hardware_link = iframe.locator("a:text('Hardware')")
    if hardware_link.count() > 0:
        return [f"click('{_bid_from_locator(hardware_link, 'Hardware link')}')"]

    return [f"goto('{_escape(task.start_url)}')"]


def cheat_custom_update_private_task(task, page, chat_messages=None, subtask_idx=None):
    if page is None:
        raise RuntimeError("UpdatePrivateTask requires a Playwright page")
    iframe = page.frame(name="gsft_main")
    if iframe is None:
        raise RuntimeError("Could not find gsft_main iframe for private task")

    search_field = iframe.get_by_label("Search a specific field of the Tasks list")
    if search_field.count() > 0:
        actions: list[str] = []
        search_bid = _bid_from_locator(search_field, "task search field")
        actions.append(f"select_option('{search_bid}', 'number')")
        search_input = iframe.locator('input[aria-label="Search"]')
        search_input_bid = _bid_from_locator(search_input, "task search input")
        if not getattr(task, "private_task_id", None):
            raise RuntimeError("UpdatePrivateTask missing private_task_id")
        actions.append(f"fill('{search_input_bid}', '{_escape(task.private_task_id)}')")
        actions.append(f"press('{search_input_bid}', 'Enter')")
        record_link = iframe.get_by_label(f"Open record: {task.private_task_id}")
        actions.append(f"click('{_bid_from_locator(record_link, 'open private task')}')")
        actions.append("noop(1500)")
        return actions

    state_select = iframe.get_by_label("state")
    if state_select.count() == 0:
        state_select = iframe.get_by_label("State")
    state_bid = _bid_from_locator(state_select, "state select")
    option = "3" if getattr(task, "set_as_completed", True) else "7"
    update_btn = iframe.locator("#sysverb_update")
    if update_btn.count() == 0:
        update_btn = iframe.get_by_text("update")
    update_bid = _bid_from_locator(update_btn, "update button")
    return [
        f"select_option('{state_bid}', '{option}')",
        f"click('{update_bid}')",
    ]


def cheat_custom_send_chat_message(task, page=None, chat_messages=None, subtask_idx=None):
    message = getattr(task, "message", "")
    return [f"send_msg_to_user('{_escape(str(message))}')"]


def cheat_custom_delete_record(task, page, chat_messages=None, subtask_idx=None):
    if page is None:
        raise RuntimeError("DeleteRecordTask requires a Playwright page")
    page_url = getattr(page, "url", "") or ""

    record_sys_id = None
    field_name = getattr(task, "field_name", None)
    field_value = getattr(task, "field_value", None)
    table_name = getattr(task, "table_name", None)
    if field_name and field_value and table_name:
        from browsergym.workarena.api.utils import table_api_call

        result = table_api_call(
            instance=task.instance,
            table=table_name,
            params={"sysparm_query": f"{field_name}={field_value}", "sysparm_fields": "sys_id"},
        )["result"]
        if result:
            record_sys_id = result[0]["sys_id"]
    if record_sys_id is None:
        record_sys_id = getattr(task, "record_sys_id", None)
    if record_sys_id and record_sys_id in page_url:
        iframe = page.frame(name="gsft_main")
        if iframe is not None:
            delete_btn = iframe.locator("#sysverb_delete")
            if delete_btn.count() == 0:
                delete_btn = iframe.get_by_text("delete")
            if delete_btn.count() > 0:
                delete_bid = _bid_from_locator(delete_btn.first, "delete button")
                confirm = iframe.locator("button:has-text('Delete')")
                if confirm.count() == 0:
                    confirm = iframe.locator("button:has-text('OK')")
                if confirm.count() == 0:
                    confirm = iframe.locator("button:has-text('Yes')")
                actions = [f"click('{delete_bid}')"]
                if confirm.count() > 0:
                    confirm_bid = _bid_from_locator(confirm.first, "confirm delete")
                    actions.append(f"click('{confirm_bid}')")
                else:
                    actions.append("noop(1000)")
                return actions
    table_name = getattr(task, "table_name", None)
    if record_sys_id and table_name:
        from browsergym.workarena.api.utils import db_delete_from_table

        db_delete_from_table(
            instance=task.instance,
            table=table_name,
            sys_id=record_sys_id,
        )

    return ["noop(1000)"]


def cheat_custom_infeasible_compositional(task, page=None, chat_messages=None, subtask_idx=None):
    if subtask_idx is None:
        raise RuntimeError("cheat_custom_infeasible_compositional requires subtask_idx")
    subtasks = getattr(task, "subtasks", None)
    if not subtasks:
        raise RuntimeError("Infeasible compositional task has no subtasks")
    cheat_index = len(subtasks) - 1 if getattr(task, "level", 2) == 2 else len(subtasks) - 3
    if subtask_idx == cheat_index:
        reason = ""
        if getattr(task, "provide_reason", False):
            reasons = getattr(task, "infeasible_reasons", None) or []
            reason = ", ".join([str(r) for r in reasons if r is not None])
        return [f"report_infeasible('{_escape(reason)}')"]
    subtask = subtasks[subtask_idx]
    ensure_cheat_custom(subtask)
    return subtask.cheat_custom(page, chat_messages)


def cheat_custom_compositional(task, page=None, chat_messages=None, subtask_idx=None):
    if subtask_idx is None:
        raise RuntimeError("cheat_custom_compositional requires subtask_idx")
    subtasks = getattr(task, "subtasks", None)
    if not subtasks:
        raise RuntimeError("Compositional task has no subtasks")
    if subtask_idx < 0 or subtask_idx >= len(subtasks):
        raise RuntimeError(f"Invalid subtask_idx {subtask_idx} (len={len(subtasks)})")
    subtask = subtasks[subtask_idx]
    ensure_cheat_custom(subtask)
    cheat_fn = getattr(subtask, "cheat_custom", None)
    if cheat_fn is None:
        raise RuntimeError(f"Subtask {type(subtask).__name__} missing cheat_custom")
    return cheat_fn(page, chat_messages)


def _register_task_id(task_id: str, fn) -> None:
    try:
        import browsergym.workarena as wa
    except Exception as exc:
        logger.warning("Could not import WorkArena for task lookup: %s", exc)
        return
    for task_cls in getattr(wa, "ALL_WORKARENA_TASKS", []):
        try:
            if getattr(task_cls, "get_task_id", None) and task_cls.get_task_id() == task_id:
                register_cheat_custom(task_cls, fn)
                return
        except Exception:
            continue
    logger.warning("Could not find WorkArena task class for %s", task_id)


def register_workarena_cheat_customs() -> None:
    try:
        from browsergym.workarena.tasks.navigation import AllMenuTask
        from browsergym.workarena.tasks.list import FilterIncidentListTask
        from browsergym.workarena.tasks.list import FilterListTask
        from browsergym.workarena.tasks.form import CreateIncidentTask
        from browsergym.workarena.tasks.service_catalog import OrderAppleWatchTask
        from browsergym.workarena.tasks.compositional.update_task import UpdatePrivateTask
        from browsergym.workarena.tasks.compositional.delete_record import (
            DeleteRecordTask,
            DeleteExpenseLineExpenseManagementTask,
            DeleteExpenseLineKnapsack,
        )
        from browsergym.workarena.tasks.send_chat_message import (
            SendChatMessageForBudgetAllocationTask,
        )
    except Exception as exc:
        logger.warning("Could not import WorkArena tasks: %s", exc)
        return

    register_cheat_custom(AllMenuTask, cheat_custom_all_menu)
    register_cheat_custom(FilterIncidentListTask, cheat_custom_filter_incident_list)
    register_cheat_custom(FilterListTask, cheat_custom_filter_list)
    register_cheat_custom(CreateIncidentTask, cheat_custom_create_incident)
    register_cheat_custom(OrderAppleWatchTask, cheat_custom_order_apple_watch)
    register_cheat_custom(UpdatePrivateTask, cheat_custom_update_private_task)
    register_cheat_custom(DeleteRecordTask, cheat_custom_delete_record)
    register_cheat_custom(DeleteExpenseLineExpenseManagementTask, cheat_custom_delete_record)
    register_cheat_custom(DeleteExpenseLineKnapsack, cheat_custom_delete_record)
    register_cheat_custom(SendChatMessageForBudgetAllocationTask, cheat_custom_send_chat_message)

    # L3 compositional tasks (delegate to subtasks)
    _register_task_id(
        "workarena.servicenow.navigate-and-create-incident-l3", cheat_custom_compositional
    )
    _register_task_id(
        "workarena.servicenow.navigate-and-filter-incident-list-l3", cheat_custom_compositional
    )
    _register_task_id(
        "workarena.servicenow.navigate-and-order-apple-watch-l3", cheat_custom_compositional
    )

    # L2 infeasible navigate-and-do tasks
    for task_id in [
        "workarena.servicenow.infeasible-navigate-and-create-change-request-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-create-hardware-asset-l2",
        "workarena.servicenow.infeasible-navigate-and-create-hardware-asset-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-create-incident-l2",
        "workarena.servicenow.infeasible-navigate-and-create-problem-l2",
        "workarena.servicenow.infeasible-navigate-and-create-user-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-asset-list-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-asset-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-change-request-list-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-change-request-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-hardware-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-incident-list-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-user-list-l2",
        "workarena.servicenow.infeasible-navigate-and-order-apple-watch-l2",
        "workarena.servicenow.infeasible-navigate-and-order-developer-laptop-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-order-ipad-mini-l2",
        "workarena.servicenow.infeasible-navigate-and-order-ipad-mini-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-order-ipad-pro-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-order-loaner-laptop-l2",
        "workarena.servicenow.infeasible-navigate-and-order-standard-laptop-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-asset-list-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-asset-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-hardware-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-incident-list-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-incident-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-service-catalog-item-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-user-list-l2",
    ]:
        _register_task_id(task_id, cheat_custom_infeasible_compositional)

    # L2 expense management tasks (delegate to subtasks)
    for task_id in [
        "workarena.servicenow.amount-based-expense-management-large-l2",
        "workarena.servicenow.amount-based-expense-management-medium-l2",
        "workarena.servicenow.basic-expense-management-large-l2",
        "workarena.servicenow.basic-expense-management-medium-l2",
        "workarena.servicenow.basic-expense-management-small-l2",
        "workarena.servicenow.date-based-expense-management-large-l2",
        "workarena.servicenow.date-based-expense-management-medium-l2",
        "workarena.servicenow.date-based-expense-management-small-l2",
        "workarena.servicenow.easy-expense-management-large-l2",
        "workarena.servicenow.easy-expense-management-medium-l2",
        "workarena.servicenow.easy-expense-management-small-l2",
    ]:
        _register_task_id(task_id, cheat_custom_compositional)

    # L2 maximize investment return tasks (delegate to subtasks)
    for task_id in [
        "workarena.servicenow.filter-random-expenses-and-delete-wrong-investments-medium-l2",
        "workarena.servicenow.filter-random-expenses-and-find-total-return-large-l2",
        "workarena.servicenow.filter-random-expenses-and-find-total-return-medium-l2",
        "workarena.servicenow.filter-random-expenses-and-find-total-return-small-l2",
        "workarena.servicenow.filter-random-expenses-and-select-investments-large-l2",
        "workarena.servicenow.filter-random-expenses-and-select-investments-medium-l2",
        "workarena.servicenow.filter-random-expenses-and-select-investments-small-l2",
        "workarena.servicenow.filter-random-expenses-find-total-return-and-select-investments-medium-l2",
        "workarena.servicenow.filter-single-item-expenses-and-delete-wrong-investments-large-l2",
        "workarena.servicenow.filter-single-item-expenses-and-delete-wrong-investments-medium-l2",
        "workarena.servicenow.filter-single-item-expenses-and-find-total-return-large-l2",
        "workarena.servicenow.filter-single-item-expenses-and-find-total-return-medium-l2",
        "workarena.servicenow.filter-single-item-expenses-and-find-total-return-small-l2",
        "workarena.servicenow.filter-single-item-expenses-and-select-investments-medium-l2",
        "workarena.servicenow.filter-single-item-expenses-find-total-return-and-select-investments-medium-l2",
        "workarena.servicenow.filter-single-item-uniform-expenses-and-delete-wrong-investments-small-l2",
        "workarena.servicenow.filter-single-item-uniform-expenses-and-select-investments-large-l2",
        "workarena.servicenow.filter-single-item-uniform-expenses-and-select-investments-medium-l2",
        "workarena.servicenow.filter-single-item-uniform-expenses-find-total-return-and-select-investments-medium-l2",
        "workarena.servicenow.filter-three-items-uniform-expenses-and-select-investments-medium-l2",
        "workarena.servicenow.filter-three-items-uniform-expenses-find-total-return-and-select-investments-large-l2",
        "workarena.servicenow.filter-three-items-uniform-expenses-find-total-return-and-select-investments-medium-l2",
        "workarena.servicenow.filter-trivial-expenses-and-find-total-return-large-l2",
        "workarena.servicenow.filter-trivial-expenses-and-select-investments-large-l2",
        "workarena.servicenow.filter-trivial-expenses-find-total-return-and-select-investments-large-l2",
        "workarena.servicenow.filter-trivial-expenses-find-total-return-and-select-investments-small-l2",
        "workarena.servicenow.filter-two-items-uniform-expenses-and-select-investments-small-l2",
    ]:
        _register_task_id(task_id, cheat_custom_compositional)
