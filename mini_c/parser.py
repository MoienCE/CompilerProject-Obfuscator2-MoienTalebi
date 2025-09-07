# === lexer_parser/parser.py ===
import ply.yacc as yacc
from .lexer import tokens
from .ast_nodes import *

# Start rule
def p_program(p):
    'program : function_list'
    p[0] = Program(p[1])

def p_function_list(p):
    '''function_list : function_list function
                     | function'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_function(p):
    'function : INT ID LPAREN param_list RPAREN LBRACE stmt_list RBRACE'
    p[0] = Function(p[2], p[4], Compound(p[7]))

def p_param_list(p):
    '''param_list : param_list COMMA param
                  | param
                  | empty'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    elif len(p) == 2 and p[1] is not None:
        p[0] = [p[1]]
    else:
        p[0] = []

def p_param(p):
    'param : INT ID'
    p[0] = (p[1], p[2])

def p_stmt_list(p):
    '''stmt_list : stmt_list statement
                 | statement'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_statement(p):
    '''statement : var_decl SEMI
                 | assignment SEMI
                 | if_stmt
                 | while_stmt
                 | return_stmt SEMI'''
    p[0] = p[1]

def p_var_decl(p):
    'var_decl : INT ID'
    p[0] = VarDecl(p[1], p[2])

def p_assignment(p):
    'assignment : ID ASSIGN expr'
    p[0] = Assignment(p[1], p[3])

def p_if_stmt(p):
    '''if_stmt : IF LPAREN expr RPAREN LBRACE stmt_list RBRACE
               | IF LPAREN expr RPAREN LBRACE stmt_list RBRACE ELSE LBRACE stmt_list RBRACE'''
    if len(p) == 8:
        p[0] = If(p[3], Compound(p[6]))
    else:
        p[0] = If(p[3], Compound(p[6]), Compound(p[10]))

def p_while_stmt(p):
    'while_stmt : WHILE LPAREN expr RPAREN LBRACE stmt_list RBRACE'
    p[0] = While(p[3], Compound(p[6]))

def p_return_stmt(p):
    'return_stmt : RETURN expr'
    p[0] = Return(p[2])

def p_expr_binop(p):
    '''expr : expr PLUS expr
            | expr MINUS expr
            | expr TIMES expr
            | expr DIVIDE expr'''
    p[0] = BinOp(p[1], p[2], p[3])

def p_expr_group(p):
    'expr : LPAREN expr RPAREN'
    p[0] = p[2]

def p_expr_number(p):
    'expr : NUMBER'
    p[0] = Num(p[1])

def p_expr_var(p):
    'expr : ID'
    p[0] = Var(p[1])

def p_expr_call(p):
    'expr : ID LPAREN arg_list RPAREN'
    p[0] = Call(p[1], p[3])

def p_arg_list(p):
    '''arg_list : arg_list COMMA expr
                | expr
                | empty'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    elif len(p) == 2 and p[1] is not None:
        p[0] = [p[1]]
    else:
        p[0] = []

def p_empty(p):
    'empty :'
    pass

def p_error(p):
    print(f"Syntax error at '{p.value}'")

parser = yacc.yacc()