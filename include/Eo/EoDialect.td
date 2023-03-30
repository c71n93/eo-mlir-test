#ifndef EO_DIALECT
#define EO_DIALECT

include "mlir/IR/OpBase.td"

/**
 * Definition of EO dialect
 */

def EO_Dialect : Dialect {
    let name = "EO";
    let summary = "A Eo out-of-tree MLIR dialect.";
    let description = [{
        This is a dialect to test how MLIR can be integraited into EO
    }];
    let cppNamespace = "::Eo";
}

/**
 * Definition of base EO operation
 */

class EO_Op<string mnemonic, list<Trait> traits = []> :
    Op<EO_Dialect, mnemonic, traits>;

#endif //EO_DIALECT
