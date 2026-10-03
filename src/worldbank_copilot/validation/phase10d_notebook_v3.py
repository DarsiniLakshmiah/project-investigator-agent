"""Phase 10D wrapper identity @3; accepted Phase 10C @2 remains immutable.

Reuse the historical parser/payload. Normalize only the complete adjacent 10D
setup block immediately after bootstrap, with its exact import at either end.
"""

from __future__ import annotations

import ast
import hashlib
import json

from worldbank_copilot.validation import phase10c_evidence as accepted

SCHEME = "databricks_wrapper_ast@3"
_IMPORT = {
    "python_ast": accepted._ast_semantics(
        ast.parse(
            "from worldbank_copilot.validation.phase10d_models import run_databricks_validation"
        ).body[0]
    )
}
_WIDGET_TARGET = accepted._ast_semantics(ast.parse("dbutils.widgets.text").body[0].value)
_NAMES = ("commit_sha", "run_id", "endpoints")


def _widget_call(entry):
    node = entry.get("python_ast", {})
    if node.get("node") != "Expr":
        return None
    call = node["fields"]["value"]
    if call.get("node") != "Call" or call["fields"]["func"] != _WIDGET_TARGET:
        return None
    return call["fields"]


def _inert_widget(entry, name):
    call = _widget_call(entry)
    if call is None or call.get("keywords") or len(call.get("args", [])) != 3:
        return False
    args = call["args"]
    return all(
        arg.get("node") == "Constant" and isinstance(arg["fields"].get("value"), str)
        for arg in args
    ) and (args[0]["fields"]["value"] == name)


def notebook_program(source: str) -> list:
    program = accepted.notebook_program(source)
    for index, entry in enumerate(program):
        if entry != {"directive": ["%run", "./_bootstrap"]}:
            continue
        start = index + 1
        block = program[start : start + 4]
        if len(block) != 4:
            continue
        # An adjoining extra widget is not an approved setup signature.
        if start + 4 < len(program) and _widget_call(program[start + 4]) is not None:
            continue
        if block[0] == _IMPORT:
            widgets = block[1:]
        elif block[-1] == _IMPORT:
            widgets = block[:-1]
        else:
            continue
        if all(_inert_widget(widget, name) for widget, name in zip(widgets, _NAMES, strict=True)):
            program[start : start + 4] = [_IMPORT, *widgets]
    return program


def notebook_identity(source: str) -> dict:
    payload = json.dumps(notebook_program(source), sort_keys=True, separators=(",", ":")).encode()
    return {"scheme": SCHEME, "sha256": hashlib.sha256(payload).hexdigest()}
