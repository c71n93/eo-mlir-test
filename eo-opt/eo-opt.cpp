#include "Eo/EoDialect.h"
#include "Eo/EoOps.h"

#include "llvm/Support/CommandLine.h"

#include "pugixml.hpp"


using namespace eo;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilePath(cl::Positional,
                                          cl::desc("<input file path>"),
                                          cl::init("-"),
                                          cl::value_desc("input file path"));

void traverseObjectsRecursive(pugi::xml_node objects) {
    static int nCalls = 0;
    nCalls++;
    for (int i = 1; i < nCalls; i++) {
        llvm::outs() << "  ";
    }
    for (auto obj: objects.children()) {
        if (!obj.attribute("name").empty()) {
            llvm::outs() << obj.attribute("name").value() << " ";
        }
        else {
            llvm::outs() << obj.attribute("base").value() << " ";
        }

        if(!obj.child("o").empty()) {
            llvm::outs() << "\n";
            traverseObjectsRecursive(obj);
            llvm::outs() << "\n";
            for (int i = 1; i < nCalls; i++) {
                llvm::outs() << "  ";
            }
        }
    }
}

int main(int argc, char* argv[]) {
    cl::ParseCommandLineOptions(argc, argv);

    pugi::xml_document doc;
    pugi::xml_parse_result res = doc.load_file(inputFilePath.c_str());
    if (!res)
        llvm::errs() << "Can't load XML file. " << res.description() << ".\n";

    traverseObjectsRecursive(doc.child("program").child("objects"));

    return 0;
}
