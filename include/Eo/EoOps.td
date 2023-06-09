#ifndef EO_OPS
#define EO_OPS

include "EoDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/**
 * Defining a ConstantOp operation. The constant operation is marked as 'Pure'
 * as it is a pure operation (don't have side effects) and may be removed if
 * dead.
 */
def ConstantOp : Eo_Op<"const", [/*ConstantLike,*/ Pure]> {
    /** `summary` and `description` provides documentation for dialect */
    let summary = "Defines a EO constant";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute. For example:

        ```mlir
        %0 = eo.const 42 : i64
        ```
    }];

    /** The `const` operation takes an attribute as the only input. */
    let arguments = (ins I64Attr:$value);

    /** The `const` operation returns a single value of I64 type. */
    let results = (outs I64:$res);
}

/**
 * Defining a ConstantOp operation.
 */
def PrintOp : Eo_Op<"print"> {
    /** `summary` and `description` provides documentation for dialect */
    let summary = "print operation";
    let description = [{
        The `eo.print` operation prints a given input integer, and produces no
        results.

        ```mlir
        %0 = eo.print 42 : i64
        ```
    }];

    /** The print operation takes an input i64 to print. */
    let arguments = (ins I64:$input);
}

#endif // EO_OPS
