## Python Guidelines
- Everything except for variables should have type annotations
- Use `uv`. Don't use `pip3` or `python3`.

## "Compute Unit" Guideline
This is a strict guideline meant to make the codebase cleaner and easier to read

__RULE__: Each statement must have at most one compute unit of each type

Types of compute units:
- Math operations: +, -, *, /, %
- Comparison operators: >, <, >=, <=
- Boolean operations: and, or
- Variable/property assignment operations: x = y
- Function/constructors calls: f(x)
- Closures: (x) => y
- Type conversions: x as y

Not a compute unit:
- Square bracket indexing
- Dot notation

What is a statement:
- In Rust and Dart, a return expression or anything that ends with a semicolon
- In Python, anything that ends with a new line, excluding multiline strings and line breaks

Counting Exceptions:
- Any math operation adding or subtracting 1 or 1.0 doesn't count
- Same consecutive math operations (two +s or two -s) count as one
- Equals (==) and not equals (!=) don't count
- The not boolean operator doesn't count
- A function call with no arguments doesn't count
- Macros/decorators and type declarations/annotations don't count