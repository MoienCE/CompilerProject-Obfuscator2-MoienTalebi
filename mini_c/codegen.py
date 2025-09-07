# === codegen.py ===

from lexer_parser.ast_nodes import *


class CodeGenerator:
    def __init__(self):
        self.output = []

    def generate(self, node):
        self.output = []
        self._gen(node)
        return '\n'.join(self.output)

    def _gen(self, node, indent=0):
        space = '    ' * indent

        if isinstance(node, Program):
            for func in node.functions:
                self._gen(func)

        elif isinstance(node, Function):
            self.output.append(f"int {node.name}({', '.join(['int ' + name for _, name in node.params])}) {{")
            self._gen(node.body, indent + 1)
            self.output.append("}")

        elif isinstance(node, Compound):
            for stmt in node.statements:
                self._gen(stmt, indent)

        elif isinstance(node, VarDecl):
            self.output.append(f"{space}int {node.name};")

        elif isinstance(node, Assignment):
            self.output.append(f"{space}{node.name} = {self._expr(node.expr)};")

        elif isinstance(node, If):
            cond = self._expr(node.condition)
            self.output.append(f"{space}if ({cond}) {{")
            self._gen(node.then_branch, indent + 1)
            self.output.append(f"{space}}}")
            if node.else_branch:
                self.output.append(f"{space}else {{")
                self._gen(node.else_branch, indent + 1)
                self.output.append(f"{space}}}")

        elif isinstance(node, While):
            cond = self._expr(node.condition)
            self.output.append(f"{space}while ({cond}) {{")
            self._gen(node.body, indent + 1)
            self.output.append(f"{space}}}")

        elif isinstance(node, Return):
            expr = self._expr(node.expr)
            self.output.append(f"{space}return {expr};")

    def _expr(self, node):
        if isinstance(node, BinOp):
            return f"({self._expr(node.left)} {node.op} {self._expr(node.right)})"
        elif isinstance(node, Num):
            return str(node.value)
        elif isinstance(node, Var):
            return node.name
        elif isinstance(node, Call):
            return f"{node.func_name}({', '.join([self._expr(arg) for arg in node.args])})"
        else:
            return "/* unknown expr */"
