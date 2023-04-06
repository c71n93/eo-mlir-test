# EO to MLIR test repository

MLIR is a C++ framework that provides a set of libraries and tools for building domain-specific compilers.

## How to convert EO to MLIR

### 1. Implement parser
**To easily traverse the program and build a MLIR from it, we need to build AST from EO file. 
There are two ways how to do it:**
1. Implement AST for EO on C++ and use existing
[ANTLR grammar](https://github.com/objectionary/eo/blob/master/eo-parser/src/main/antlr4/org/eolang/parser/Program.g4)
to generate parser. Then easily traverse this AST and generate MLIR.
2. Use ready-made `xmir` files to travers them and generate MLIR.

*We need to decide which approach is more correct to use*

### 2. Design high-level EO dialect for MLIR
MLIR provides a powerful declaratively specification mechanism via TableGen; a generic language with tooling to maintain 
records of domain-specific information; that simplifies the definition process by automatically generating all of the 
necessary boilerplate C++ code, significantly reduces maintainence burden when changing aspects of dialect definitions, 
and also provides additional tools on top (such as documentation generation). More information 
[here](https://mlir.llvm.org/docs/DefiningDialects/).
#### Some ideas on implementation of EO dialect for MLIR
We need to design EO dialect as high-level and close to the original language as possible.

*Attention: the MLIR code presented here is not a working example, but only a prototype of the dialect*
##### 1. Object definition = function definition

The idea is based on the fact that all objects in EO are dataized into something that is copied in their phi 
attribute. So object can be represented as function, that returns phi attribute.

So far, this is a simple implementation based on the built-in 
[func](https://mlir.llvm.org/docs/Dialects/Func/#funccall-mlirfunccallop) dialect. In the future, it will be possible 
to make a custom operation, called, for example `abstraction`.

- Case 1: Object that just decorates another object

    ```
    [a b] > sum
      a.plus b > @
  
    sum 1 2 > sum_copy
    ```
  Converts to:
    ```
    func.func @sum(%a: i64, %b: i64) -> i64 {
        %phi = eo.plus(%a, %b) : (i64, i64) -> i64
        func.return %phi : i64
    }
    
    %phi = func.call @sum(1, 2) : (i64, i64) -> i64
    %sum_copy = eo.copy(%phi) : (i64) -> i64
    ```
  *Here `sum` object definition converts to `@sum` function in MLIR, that accepts two `i64` attributes and returns one 
`i64` value. In fact, we don't know what types of the object in this program, but to simplify the example, let's assume 
that we have already inferred them.*

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
    func.func @sum(%a: i64, %b: i64) -> (i64, i64, i64) {
        %phi = eo.plus(%a, %b) : (i64, i64) -> i64
        %first = eo.copy(%a) : (i64) -> i64
        %second = eo.copy(%b) : (i64) -> i64
        func.return %phi, %first, %second : i64, i64, i64
    }
    
    %phi, %first, %second = eo.call @sum(1, 2) : (i64, i64) -> (i64, i64, i64)
    %sum_copy = eo.copy(%phi) : (i64) -> i64
    %first_copy = eo.copy(%first) : (i64) -> i64
    ```

- Case 3: Nested objects (current problem)
    ```
    [a b] > sum
      a.plus b > @
      [] > args
        a > first
        b > second
    
    sum 1 2 > sum_copy
    sum.args > args_copy
    args_copy.first > first_copy
    ```
    Converts to:
    ```
    func.func @sum(%a: i64, %b: i64) -> (i64, i64, i64) {
        %phi = eo.plus(%a, %b) : (i64, i64) -> i64
        func.func @args(%a: i64, %b: i64) -> (i64, i64) { // here we should explicitly pass attributes of parent object
            %first = eo.copy(%a) : (i64) -> i64
            %second = eo.copy(%b) : (i64) -> i64
            func.return %first, %second -> i64, i64
        }
        %args = func.constant @args : (i64, i64) -> (i64, i64)
        func.return %phi, args : i64, ((i64, i64) -> (i64, i64))
    }
    
    %phi, %args = eo.call @sum(1, 2) : (i64, i64) -> (i64, i64, i64)
    %args_copy = eo.copy(%args) : ((i64, i64) -> (i64, i64)) -> ((i64, i64) -> (i64, i64))
    %first_copy = func.call_indirect %args_copy(1, 2) : (i64, i64) -> (i64, i64)
    ```
  *Here we should to explicitly pass arguments to `%args_copy` function. This is a bad decision and we need to figure 
out how to make it better*

- Case 3: Partial application (current problem)
    ```
    [a b] > sum
      a.plus b > @
      a > first
      b > second
    
    sum 1 > plus_one
    plus_one 2 > three
    ```
  *What should function return?*

##### 2. Object definition = creating a new type
This solution is most likely impossible to implement for EO. Since it is not yet known how objects can be represented 
in EO as static types.
- Case 1: Object copying (current problem)

    ```
    [] > numbers
      1 > one
      2 > two
    
    numbers > numbers_copy
    numbers.one > one
    ```
    Converts to:
    ```
    %0 = eo.copy(???) : (!eo.object<i64, i64>) -> i64 // what to copy here?
    %1 = eo.dot_notation %0[0] (!eo.object<i64, i64>) -> i64 // 
    %2 = eo.copy(%1) : (i64) -> i64 
    ```
- Case 2: Free attributes (current problem)
    ```
    [a b] > sum
      a.plus b > @
    
    numbers > numbers_copy
    numbers.one > one
    ```
    *It is impossible to pass arguments to static type*

##### 3. High level operations (version 1)
The idea is to implement dialect as close to original EO program as possible. And also make it possible to lower some
necessary operations to other dialects. Let's see how it works on some examples.

- Case 1: Object that just decorates another object
  
  ```
  [a b] > sum
    a.plus b > @
  
  [] > app
    sum 1 2 > @
  ```
  Converts to:
  ```
  eo.abstraction @sum(%a: !eo.obj, %b: !eo.obj) : (!eo.obj, !eo.obj) -> !eo.obj {
      // create empty object
      %sum = eo.create_empty_obj() : () -> !eo.obj
      
      // [a b]
      %sum1 = eo.add_attr(%sum, %a) {attr = "a"} : (!eo.obj, !eo.obj) -> !eo.obj
      %sum2 = eo.add_attr(%sum1, %b) {attr = "b"} : (!eo.obj, !eo.obj) -> !eo.obj
      
      // a.plus b > @
      %a_plus = eo.dot_notation(%a) {attr = "plus"} : (!eo.obj) -> (!eo.obj)
      %a_plus_b = eo.application(%a_plus, [%b]) : (!eo.obj, !eo.obj[]) -> !eo.obj
      %sum3 = eo.add_attr(%sum2, %a_plus_b) {attr = "phi"} : (!eo.obj, !eo.obj) -> !eo.obj
  
      return %sum3
  }
  
  eo.abstraction @app() : () -> !eo.obj {
      // create empty object
      %app = eo.create_empty_obj() : () -> !eo.obj
  
      // sum(1 2).first > @
      %sum_abstr = eo.get_obj() {obj = @sum} : ((!eo.obj, !eo.obj) -> !eo.obj) -> !eo.obj
      %1 = eo.get_obj() {obj = 1} :i64 -> !eo.obj
      %2 = eo.get_obj() {obj = 2} :i64 -> !eo.obj
      %sum_obj = %a_plus_b = eo.application(%sum_abstr, [%1, %2]) : (!eo.obj, !eo.obj[]) -> !eo.obj
      %app1 = eo.add_attr(%app, %sum_obj) {attr = "phi"} : (!eo.obj, !eo.obj) -> !eo.obj
      
      return %app1
  }
  ```
  *Here `!eo.obj` custom type that is map of object attributes(which are objects in themselves) and their names.*
- Case 2: Object that has some attributes

  ```
  [a b] > sum
    a.plus b > @
    a > first
    b > second
  
  [] > app
    first. > @
      sum 1 2
  ```
  Converts to:

  ```
  eo.abstraction @sum(%a: !eo.obj, %b: !eo.obj) : (!eo.obj, !eo.obj) -> !eo.obj {
      // create empty object
      %sum = eo.create_empty_obj() : () -> !eo.obj
      
      // [a b]
      %sum1 = eo.add_attr(%sum, %a) {attr = "a"} : (!eo.obj, !eo.obj) -> !eo.obj
      %sum2 = eo.add_attr(%sum1, %b) {attr = "b"} : (!eo.obj, !eo.obj) -> !eo.obj
      
      // a.plus b > @
      %a_plus = eo.dot_notation(%a) {attr = "plus"} : (!eo.obj) -> (!eo.obj)
      %a_plus_b = eo.application(%a_plus, [%b]) : (!eo.obj, !eo.obj[])
      %sum3 = eo.add_attr(%sum2, %a_plus_b) {attr = "phi"} : (!eo.obj, !eo.obj) -> !eo.obj
  
      // a > first
      %sum4 = eo.add_attr(%sum3, %a) {attr = "first"} : (!eo.obj, !eo.obj) -> !eo.obj
  
      // b > second
      %sum5 = eo.add_attr(%sum4, %b) {attr = "second"} : (!eo.obj, !eo.obj) -> !eo.obj
  
      return %sum5
  }
  
  eo.abstraction @app() : () -> !eo.obj {
      // create empty object
      %app = eo.create_empty_obj() : () -> !eo.obj
  
      // sum(1 2).first > @
      %sum_abstr = eo.get_obj() {obj = @sum} : ((!eo.obj, !eo.obj) -> !eo.obj) -> !eo.obj
      %1 = eo.get_obj() {obj = 1} :i64 -> !eo.obj
      %2 = eo.get_obj() {obj = 2} :i64 -> !eo.obj
      %sum_obj = %a_plus_b = eo.application(%sum_abstr, [%1, %2]) : (!eo.obj, !eo.obj[]) -> !eo.obj
      %app1 = eo.add_attr(%app, %sum_first) {attr = "phi"} : (!eo.obj, !eo.obj) -> !eo.obj
      
      return %app1
  }
  ```


### 3. Implement conversion from EO AST to MLIR EO dialect
Using [Builder](https://mlir.llvm.org/doxygen/classmlir_1_1Builder.html) class implement `EoMLIRGen` class, that will
traverse AST and build MLIR file from EO program. 

### 4. Figure out how to get CFG from MLIR
Now that we have a high-level EO dialect for MLIR, we can [lower](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/) it to 
some built-in dialect, that can be represented as CFG. For example we can lower EO dialict to the 
[cf](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/#cfassert-mlircfassertop) dialect.

*We need to decide to which dialect, or combination of dialects we need to lower to get CFG*