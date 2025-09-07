# === lexer_parser/ast_nodes.py ===
class Node:
    pass

class Program(Node):
    def __init__(self, functions):
        self.functions = functions

class Function(Node):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class Compound(Node):
    def __init__(self, statements):
        self.statements = statements

class VarDecl(Node):
    def __init__(self, vtype, name):
        self.vtype = vtype
        self.name = name

class Assignment(Node):
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

class If(Node):
    def __init__(self, condition, then_branch, else_branch=None):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

class While(Node):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class Return(Node):
    def __init__(self, expr):
        self.expr = expr

class BinOp(Node):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class Num(Node):
    def __init__(self, value):
        self.value = value

class Var(Node):
    def __init__(self, name):
        self.name = name

class Call(Node):
    def __init__(self, func_name, args):
        self.func_name = func_name
        self.args = args