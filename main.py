# main.py
# Minimal Mini-C deobfuscator for Phase 2 — single-file, no external deps
# Features: expr simplify, dead-code elim, opaque-if removal, block tidy, rename vars

import re
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, Set

# ──────────────────────────────────────────────────────────────────────────────
# Preprocess: remove BOM, normalize newlines, strip comments
def preprocess(src: str) -> str:
    # remove UTF-8 BOM if present (redundant with utf-8-sig, but harmless)
    src = src.replace("\ufeff", "")
    # normalize newlines
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    # remove zero-width and other invisible troublemakers
    src = re.sub(r"[\u200B\u200C\u200D\u2060\uFEFF]", "", src)
    # strip // line comments
    src = re.sub(r"//[^\n]*", "", src)
    # strip /* ... */ block comments
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return src

# ──────────────────────────────────────────────────────────────────────────────
# Lexer (tiny)
TOKEN_SPEC = [
    ("INT",      r"\bint\b"),
    ("RETURN",   r"\breturn\b"),
    ("IF",       r"\bif\b"),
    ("ELSE",     r"\belse\b"),
    ("ID",       r"[A-Za-z_][A-Za-z0-9_]*"),
    ("NUMBER",   r"\d+"),
    ("EQ",       r"=="),
    ("ASSIGN",   r"="),
    ("PLUS",     r"\+"),
    ("MINUS",    r"-"),
    ("STAR",     r"\*"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("LBRACE",   r"\{"),
    ("RBRACE",   r"\}"),
    ("SEMI",     r";"),
    ("WS",       r"[ \t\r\n]+"),
    ("OTHER",    r"."),  # anything else → error
]
MASTER = re.compile("|".join(f"(?P<{n}>{p})" for n,p in TOKEN_SPEC))

@dataclass
class Tok:
    kind: str
    text: str
    pos: int


def lex(src: str) -> list:
    toks = []
    for m in MASTER.finditer(src):
        kind = m.lastgroup
        text = m.group()
        if kind == "WS":
            continue
        if kind == "OTHER":
            ch = text
            code = ord(ch)
            raise SyntaxError(f"Unexpected char `{ch}` (U+{code:04X}) at {m.start()}")
        toks.append(Tok(kind, text, m.start()))
    return toks

# ──────────────────────────────────────────────────────────────────────────────
# AST nodes
class Node: ...
@dataclass
class Program(Node):
    func: "Function"

@dataclass
class Function(Node):
    rettype: str
    name: str
    body: "Block"

@dataclass
class Block(Node):
    stmts: List["Stmt"] = field(default_factory=list)

class Stmt(Node): ...
@dataclass
class VarDecl(Stmt):
    name: str
    init: Optional["Expr"]=None

@dataclass
class Assign(Stmt):
    name: str
    expr: "Expr"

@dataclass
class If(Stmt):
    cond: "Expr"
    then_branch: Block
    else_branch: Optional[Block]=None

@dataclass
class Return(Stmt):
    expr: "Expr"

# Expr
class Expr(Node): ...
@dataclass
class Var(Expr):
    name: str

@dataclass
class Num(Expr):
    value: int

@dataclass
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr

@dataclass
class Unary(Expr):
    op: str
    expr: Expr

# ──────────────────────────────────────────────────────────────────────────────
# Parser (recursive descent) — supports: function with body; decl/assign/if/return
class Parser:
    def __init__(self, toks: List[Tok]):
        self.toks = toks
        self.i = 0

    def peek(self, k=0) -> Optional[Tok]:
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else None

    def eat(self, kind: str) -> Tok:
        t = self.peek()
        if t is None or t.kind != kind:
            raise SyntaxError(f"Expected {kind} at {self.peek().pos if self.peek() else 'EOF'}")
        self.i += 1
        return t

    def accept(self, kind: str) -> Optional[Tok]:
        if self.peek() and self.peek().kind == kind:
            return self.eat(kind)
        return None

    # program: 'int' ID '(' ')' '{' stmt* '}'
    def parse_program(self) -> Program:
        self.eat("INT")
        name = self.eat("ID").text
        self.eat("LPAREN"); self.eat("RPAREN")
        body = self.parse_block()
        return Program(Function("int", name, body))

    def parse_block(self) -> Block:
        self.eat("LBRACE")
        stmts = []
        while self.peek() and self.peek().kind != "RBRACE":
            stmts.append(self.parse_stmt())
        self.eat("RBRACE")
        return Block(stmts)

    def parse_stmt(self) -> Stmt:
        t = self.peek()
        if t.kind == "INT":
            self.eat("INT")
            name = self.eat("ID").text
            init = None
            if self.accept("ASSIGN"):
                init = self.parse_expr()
            self.eat("SEMI")
            return VarDecl(name, init)
        if t.kind == "ID":
            name = self.eat("ID").text
            self.eat("ASSIGN")
            expr = self.parse_expr()
            self.eat("SEMI")
            return Assign(name, expr)
        if t.kind == "IF":
            self.eat("IF")
            self.eat("LPAREN")
            cond = self.parse_expr()
            self.eat("RPAREN")
            then_b = self.parse_block()
            else_b = None
            if self.accept("ELSE"):
                else_b = self.parse_block()
            return If(cond, then_b, else_b)
        if t.kind == "RETURN":
            self.eat("RETURN")
            expr = self.parse_expr()
            self.eat("SEMI")
            return Return(expr)
        if t.kind == "LBRACE":
            return self.parse_block()
        raise SyntaxError(f"Unexpected token {t.kind} at {t.pos}")

    # expr with precedence: ==  ;  + -  ;  *  ; unary -  ;  atoms
    def parse_expr(self) -> Expr:
        e = self.parse_add()
        while self.accept("EQ"):
            rhs = self.parse_add()
            e = BinOp("==", e, rhs)
        return e

    def parse_add(self) -> Expr:
        e = self.parse_mul()
        while True:
            if self.accept("PLUS"):
                e = BinOp("+", e, self.parse_mul())
            elif self.accept("MINUS"):
                e = BinOp("-", e, self.parse_mul())
            else:
                return e

    def parse_mul(self) -> Expr:
        e = self.parse_unary()
        while self.accept("STAR"):
            e = BinOp("*", e, self.parse_unary())
        return e

    def parse_unary(self) -> Expr:
        if self.accept("MINUS"):
            return Unary("-", self.parse_unary())
        return self.parse_atom()

    def parse_atom(self) -> Expr:
        t = self.peek()
        if t.kind == "NUMBER":
            self.eat("NUMBER")
            return Num(int(t.text))
        if t.kind == "ID":
            self.eat("ID")
            return Var(t.text)
        if t.kind == "LPAREN":
            self.eat("LPAREN")
            e = self.parse_expr()
            self.eat("RPAREN")
            return e
        raise SyntaxError(f"Unexpected token {t.kind} at {t.pos}")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: traversal & utils
def collect_reads_expr(e: Expr, out: Set[str]):
    if isinstance(e, Var):
        out.add(e.name)
    elif isinstance(e, (Num,)):
        pass
    elif isinstance(e, Unary):
        collect_reads_expr(e.expr, out)
    elif isinstance(e, BinOp):
        collect_reads_expr(e.left, out); collect_reads_expr(e.right, out)

def collect_reads_stmt(s: Stmt, out: Set[str]):
    if isinstance(s, VarDecl) and s.init:
        collect_reads_expr(s.init, out)
    elif isinstance(s, Assign):
        collect_reads_expr(s.expr, out)
    elif isinstance(s, If):
        collect_reads_expr(s.cond, out)
        for st in s.then_branch.stmts:
            collect_reads_stmt(st, out)
        if s.else_branch:
            for st in s.else_branch.stmts:
                collect_reads_stmt(st, out)
    elif isinstance(s, Return):
        collect_reads_expr(s.expr, out)

def const_eval(e: Expr) -> Optional[int]:
    """Evaluate integer expression if fully constant (no Var)."""
    if isinstance(e, Num): return e.value
    if isinstance(e, Unary) and e.op == "-":
        v = const_eval(e.expr)
        return -v if v is not None else None
    if isinstance(e, BinOp):
        l = const_eval(e.left); r = const_eval(e.right)
        if l is None or r is None: return None
        if e.op == "+": return l + r
        if e.op == "-": return l - r
        if e.op == "*": return l * r
        if e.op == "==": return 1 if (l == r) else 0
    return None

# Expr simplifier rules
def simplify_expr(e: Expr) -> Expr:
    # post-order
    if isinstance(e, Unary):
        e.expr = simplify_expr(e.expr)
        # -(-x) → x
        if e.op == "-" and isinstance(e.expr, Unary) and e.expr.op == "-":
            return e.expr.expr
        # constant fold
        v = const_eval(e)
        return Num(v) if v is not None else e

    if isinstance(e, BinOp):
        e.left = simplify_expr(e.left)
        e.right = simplify_expr(e.right)

        # x - (-y) → x + y
        if e.op == "-" and isinstance(e.right, Unary) and e.right.op == "-":
            e = BinOp("+", e.left, e.right.expr)

        # +0, 0+  ;  *1, *0, 1*
        if e.op == "+":
            if isinstance(e.right, Num) and e.right.value == 0: return e.left
            if isinstance(e.left, Num) and e.left.value == 0: return e.right
        if e.op == "*":
            if isinstance(e.right, Num) and e.right.value == 1: return e.left
            if isinstance(e.left, Num) and e.left.value == 1: return e.right
            if isinstance(e.right, Num) and e.right.value == 0: return Num(0)
            if isinstance(e.left, Num) and e.left.value == 0: return Num(0)

        # constant fold
        v = const_eval(e)
        return Num(v) if v is not None else e

    return e  # Var or Num

def simplify_stmt(s: Stmt) -> Stmt:
    if isinstance(s, VarDecl):
        if s.init: s.init = simplify_expr(s.init); return s
        return s
    if isinstance(s, Assign):
        s.expr = simplify_expr(s.expr); return s
    if isinstance(s, If):
        s.cond = simplify_expr(s.cond)
        # opaque: if (1) {..} else {...}
        cv = const_eval(s.cond)
        if cv is not None:
            if cv != 0:
                # keep then branch only
                return Block([simplify_stmt(x) for x in s.then_branch.stmts])  # flatten one level
            else:
                if s.else_branch:
                    return Block([simplify_stmt(x) for x in s.else_branch.stmts])
                else:
                    return Block([])
        s.then_branch = simplify_block(s.then_branch)
        if s.else_branch: s.else_branch = simplify_block(s.else_branch)
        return s
    if isinstance(s, Return):
        s.expr = simplify_expr(s.expr); return s
    if isinstance(s, Block):
        return simplify_block(s)
    return s

def simplify_block(b: Block) -> Block:
    new = []
    for st in b.stmts:
        st2 = simplify_stmt(st)
        # flatten nested blocks
        if isinstance(st2, Block):
            new.extend(st2.stmts)
        else:
            new.append(st2)
    # remove empty statements (there are none explicit)
    return Block(new)

# Dead code elimination (vars never read)
def dead_code(block: Block) -> Block:
    # collect all reads
    reads: Set[str] = set()
    for st in block.stmts:
        collect_reads_stmt(st, reads)

    new = []
    for st in block.stmts:
        # remove pure expr statements -> we don't have standalone exprs
        if isinstance(st, VarDecl) and st.name not in reads:
            # drop decl if never read; but keep if initialized with side-effects (we assume none)
            continue
        # remove assignments to never-read vars
        if isinstance(st, Assign) and st.name not in reads:
            continue
        new.append(st)

    return Block(new)

# Rename variables: v\d+, junk*, dummy*, waste* → a,b,c,...
def rename_vars(block: Block) -> Tuple[Block, dict]:
    order: List[str] = []
    def record(name: str):
        if name not in order:
            if re.match(r"^(v\d+|junk\d*|dummy\d*|waste\d*)$", name):
                order.append(name)

    # collect
    def walk_stmt(st: Stmt):
        if isinstance(st, VarDecl):
            record(st.name)
            if st.init: walk_expr(st.init)
        elif isinstance(st, Assign):
            record(st.name); walk_expr(st.expr)
        elif isinstance(st, If):
            walk_expr(st.cond)
            for x in st.then_branch.stmts: walk_stmt(x)
            if st.else_branch:
                for x in st.else_branch.stmts: walk_stmt(x)
        elif isinstance(st, Return):
            walk_expr(st.expr)
    def walk_expr(e: Expr):
        if isinstance(e, Var): record(e.name)
        elif isinstance(e, Unary): walk_expr(e.expr)
        elif isinstance(e, BinOp): walk_expr(e.left); walk_expr(e.right)

    for st in block.stmts: walk_stmt(st)

    alphabet = [chr(c) for c in range(ord('a'), ord('z')+1)]
    names = []
    i = 0
    while len(names) < len(order):
        if i < 26:
            names.append(alphabet[i])
        else:
            names.append(alphabet[i%26] + str(i//26))
        i += 1
    mapping = dict(zip(order, names))

    def apply_stmt(st: Stmt) -> Stmt:
        def rn(name: str) -> str: return mapping.get(name, name)

        if isinstance(st, VarDecl):
            st.name = rn(st.name)
            if st.init: st.init = apply_expr(st.init)
            return st
        if isinstance(st, Assign):
            st.name = rn(st.name)
            st.expr = apply_expr(st.expr)
            return st
        if isinstance(st, If):
            st.cond = apply_expr(st.cond)
            st.then_branch = apply_block(st.then_branch)
            if st.else_branch: st.else_branch = apply_block(st.else_branch)
            return st
        if isinstance(st, Return):
            st.expr = apply_expr(st.expr)
            return st
        return st

        # nested blocks handled in apply_block

    def apply_expr(e: Expr) -> Expr:
        if isinstance(e, Var):
            return Var(mapping.get(e.name, e.name))
        if isinstance(e, Unary):
            return Unary(e.op, apply_expr(e.expr))
        if isinstance(e, BinOp):
            return BinOp(e.op, apply_expr(e.left), apply_expr(e.right))
        return e

    def apply_block(b: Block) -> Block:
        return Block([apply_stmt(s) if not isinstance(s, Block) else apply_block(s) for s in b.stmts])

    return apply_block(block), mapping

# Pretty codegen
def emit_program(p: Program) -> str:
    out = []
    out.append(f"{p.func.rettype} {p.func.name}() {{")
    body = emit_block(p.func.body, indent=4)
    out.append(body)
    out.append("}")
    return "\n".join(out)

def emit_block(b: Block, indent=0) -> str:
    sp = " " * indent
    lines = []
    for st in b.stmts:
        if isinstance(st, VarDecl):
            if st.init is None:
                lines.append(f"{sp}int {st.name};")
            else:
                lines.append(f"{sp}int {st.name} = {emit_expr(st.init)};")
        elif isinstance(st, Assign):
            lines.append(f"{sp}{st.name} = {emit_expr(st.expr)};")
        elif isinstance(st, If):
            lines.append(f"{sp}if ({emit_expr(st.cond)}) {{")
            lines.append(emit_block(st.then_branch, indent+4))
            lines.append(f"{sp}}}")
            if st.else_branch and st.else_branch.stmts:
                lines.append(f"{sp}else {{")
                lines.append(emit_block(st.else_branch, indent+4))
                lines.append(f"{sp}}}")
        elif isinstance(st, Return):
            lines.append(f"{sp}return {emit_expr(st.expr)};")
        elif isinstance(st, Block):
            lines.append(f"{sp}{{")
            lines.append(emit_block(st, indent+4))
            lines.append(f"{sp}}}")
    return "\n".join([ln for ln in lines if ln.strip() != ""])

def emit_expr(e: Expr) -> str:
    if isinstance(e, Num): return str(e.value)
    if isinstance(e, Var): return e.name
    if isinstance(e, Unary): return f"-({emit_expr(e.expr)})"
    if isinstance(e, BinOp):
        op = e.op
        return f"({emit_expr(e.left)} {op} {emit_expr(e.right)})"
    return "/*?*/"

# Pipeline
def run_pipeline(prog: Program) -> Program:
    # 1) simplify expressions + opaque-if removal + block tidy
    prog.func.body = simplify_block(prog.func.body)

    # 2) dead code elim
    prog.func.body = dead_code(prog.func.body)

    # 3) simplify again (پس از حذف‌ها فرصت ساده‌سازی بیشتر ایجاد می‌شود)
    prog.func.body = simplify_block(prog.func.body)

    # 4) rename variables (v*, junk*, dummy*, waste*)
    prog.func.body, _ = rename_vars(prog.func.body)

    # 5) یک دور نهایی تمیزکاری
    prog.func.body = simplify_block(prog.func.body)
    return prog

# Entry
def main():
    ap = argparse.ArgumentParser(description="Phase-2 Mini-C deobfuscator (single file).")
    ap.add_argument("infile", help="obfuscated C-like file (phase-1 output)")
    ap.add_argument("outfile", help="cleaned output")
    args = ap.parse_args()

    with open(args.infile, "r", encoding="utf-8-sig") as f:  # utf-8-sig removes BOM
        src = f.read()


    # <<< fix: preprocess for BOM & comments >>>
    src = preprocess(src)

    toks = lex(src)
    prog = Parser(toks).parse_program()
    prog = run_pipeline(prog)
    out = emit_program(prog)

    with open(args.outfile, "w", encoding="utf-8") as f:
        f.write(out)

    print(f"✅ Wrote cleaned code → {args.outfile}")

if __name__ == "__main__":
    main()
