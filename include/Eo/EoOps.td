#ifndef EO_OPS
#define EO_OPS

include "EoDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// TODO: Is this Operation is Pure(with no side effects)?
def PrintOp : EO_Op<"print", [Pure]> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input integer, and
        produces no results.
    }];

    // The print operation takes an input integer to print.
    let arguments = (ins I64:$input);
}

#endif // EO_OPS
