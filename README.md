# EO to MLIR test repository

MLIR is a C++ framework that provides a set of libraries and tools for building domain-specific compilers.

## How to convert EO to MLIR

### 1. Implement parser
**To easily traverse the program and build a MLIR from it, we need to build some sort of AST from EO file. 
There are two ways how to do it:**
1. Implement AST for EO on C++ and use existing
[ANTLR grammar](https://github.com/objectionary/eo/blob/master/eo-parser/src/main/antlr4/org/eolang/parser/Program.g4)
to generate parser. Then easily traverse this AST and generate MLIR.
2. Use ready-made `xmir` files to travers them and generate MLIR.

### 2. Design high-level EO dialect for MLIR
MLIR provides a powerful declaratively specification mechanism via TableGen; a generic language with tooling to maintain 
records of domain-specific information; that simplifies the definition process by automatically generating all of the 
necessary boilerplate C++ code, significantly reduces maintainence burden when changing aspects of dialect definitions, 
and also provides additional tools on top (such as documentation generation). More information 
[here](https://mlir.llvm.org/docs/DefiningDialects/).
#### Some ideas on implementation of EO dialect for MLIR
*Attention: the MLIR code presented here is not a working example, but only a prototype of the dialect*
##### 1. Object definition = function declaration
- Case 1: Object that just decorates another object

    ```
    [a b] > sum
      a.plus b > @
    ```
    Converts to:
    ```
    eo.func @sum(%arg1: i64, %arg2: i64) -> i64 {
        %0 = eo.plus(%arg1, %arg2) : (i64, i64) -> i64
        eo.return(%0) : (i64) -> ()
    }
    ```
  
- Case 2: Object that has some attributes
    ```
    [a b] > sum
      a.plus b > @
      a > first
      b > second
    
    sum 1 2 > sum_copy
    sum.first > first_copy
    ```
  Converts to:
    ```
    eo.func @sum(%arg1: i64, %arg2: i64) -> (i64, i64, i64) {
        %0 = eo.plus(%arg1, %arg2) : (i64, i64) -> i64
        %1 = eo.copy(%arg1) : (i64) -> i64
        %2 = eo.copy(%arg2) : (i64) -> i64
        eo.return(%0, %1, %2) : (i64, i64, i64) -> ()
    }
    
    %0, %1, %2 = eo.call @sum(1, 2) : (i64, i64) -> (i64, i64, i64)
    %3 = eo.copy(%0) : (i64) -> i64 // sum_copy
    %3 = eo.copy(%1) : (i64) -> i64 // first_copy
    ```
- Case 3: Partial application (current problem)
    ```
    [a b] > sum
      a.plus b > @
      a > first
      b > second
    
    sum 1 > plus_one
    plus_one 2 > three
    ```
    Converts to ???

##### 1. Object definition = creating a new type
- Case 1: Object copying:
    ```
    [] > numbers
      1 > one
      2 > two
    
    numbers > numbers_copy
    numbers.one > one
    ```
    Converts to:
    ```
    %0 = eo.copy(???) : (!eo.object<i64, i64>) -> i64 // what to copy?
    %1 = eo.dot_notation %0[0] (!eo.object<i64, i64>) -> i64
    %2 = eo.copy(%1) : (i64) -> i64 
    ```
- Case 2: Free attributes (current problem)
    ```
    [a b] > sum
      a.plus b > @
    
    numbers > numbers_copy
    numbers.one > one
    ```
    Converts to ???

### 3. Implement conversion from EO AST to MLIR EO dialect
Using [Builder](https://mlir.llvm.org/doxygen/classmlir_1_1Builder.html) class implement `EoMLIRGen` class, that will
traverse AST and build MLIR file from EO program.

### 4. Figure out how to get CFG from MLIR
Current problem: how to get CFG from MLIR?
From what level of representation in MLIR is it already possible to get CFG for EO?
- Directly from the EO dialect
- Lower to the dialect [cf](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/#cfassert-mlircfassertop )
- From any combination of dialects for which lowering is defined to the LLVM IR dialect
