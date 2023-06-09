#ifndef EO_DIALECT
#define EO_DIALECT

include "mlir/IR/OpBase.td"

/**
 * Definition of EO dialect
 */
def Eo_Dialect : Dialect {
    let name = "eo";
    let summary = "A Eo out-of-tree MLIR dialect.";
    let description = [{
        This is a dialect to test how MLIR can be integraited into EO
    }];
    let cppNamespace = "::eo";
}

/**
 * Definition of base EO operation
 */
class Eo_Op<string mnemonic, list<Trait> traits = []> :
    Op<Eo_Dialect, mnemonic, traits>;

#endif //EO_DIALECT
