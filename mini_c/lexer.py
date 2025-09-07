# === lexer_parser/lexer.py ===
import ply.lex as lex

# Token definitions
tokens = (
    'ID', 'NUMBER', 'CHAR',
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
    'SEMI', 'COMMA', 'ASSIGN',
    'EQ', 'NEQ', 'LT', 'GT', 'LEQ', 'GEQ',
    'IF', 'ELSE', 'WHILE', 'FOR', 'RETURN',
    'INT', 'CHAR_TYPE', 'BOOL', 'PRINTF', 'SCANF'
)

# Reserved keywords
reserved = {
    'if': 'IF', 'else': 'ELSE', 'while': 'WHILE',
    'for': 'FOR', 'return': 'RETURN', 'int': 'INT',
    'char': 'CHAR_TYPE', 'bool': 'BOOL', 'printf': 'PRINTF', 'scanf': 'SCANF'
}

# Token regex

# Symbols
t_PLUS    = r'\+'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_LBRACE  = r'\{'
t_RBRACE  = r'\}'
t_SEMI    = r';'
t_COMMA   = r','
t_ASSIGN  = r'='
t_EQ      = r'=='
t_NEQ     = r'!='
t_LT      = r'<'
t_GT      = r'>'
t_LEQ     = r'<='
t_GEQ     = r'>='

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'ID')
    return t

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_CHAR(t):
    r"'[^']'"
    t.value = t.value[1]
    return t

t_ignore = ' \t'

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_comment(t):
    r'//.*'
    pass

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()
