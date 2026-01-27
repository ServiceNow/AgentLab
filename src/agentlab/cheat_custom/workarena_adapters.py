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
    kind = task.filter_kind
    list_info = task.list_info

    predicates = []
    for col, val in zip(columns, values):
        if val is None or val == "":
            predicates.append(f"{col}ISEMPTY")
            continue

        col_info = list_info["columns"][col]
        col_type = col_info.get("type")

        if col_type == "choice":
            choices = col_info.get("choices", {})
            internal = None
            for key, display in choices.items():
                if display == val:
                    internal = key
                    break
            if internal is None:
                internal = val
            predicates.append(f"{col}={internal}")
            continue

        if col_type == "reference":
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
                    predicates.append(f"{col}={res[0]['sys_id']}")
                    continue
            predicates.append(f"{col}={val}")
            continue

        predicates.append(f"{col}={val}")

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
        from browsergym.workarena.tasks.form import CreateIncidentTask
        from browsergym.workarena.tasks.service_catalog import OrderAppleWatchTask
        from browsergym.workarena.tasks.compositional.update_task import UpdatePrivateTask
    except Exception as exc:
        logger.warning("Could not import WorkArena tasks: %s", exc)
        return

    register_cheat_custom(AllMenuTask, cheat_custom_all_menu)
    register_cheat_custom(FilterIncidentListTask, cheat_custom_filter_incident_list)
    register_cheat_custom(CreateIncidentTask, cheat_custom_create_incident)
    register_cheat_custom(OrderAppleWatchTask, cheat_custom_order_apple_watch)
    register_cheat_custom(UpdatePrivateTask, cheat_custom_update_private_task)

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
