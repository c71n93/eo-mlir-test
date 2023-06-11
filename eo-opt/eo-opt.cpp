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
void traverseObjects(pugi::xml_node objects) {

    for (auto obj: objects.children()) {
        llvm::outs() << obj.attribute("name").value() << "\n";
        for (auto sub_obj: obj.children()) {
            llvm::outs() << "\t" << sub_obj.attribute("name").value() << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    cl::ParseCommandLineOptions(argc, argv);

    pugi::xml_document doc;
    pugi::xml_parse_result res = doc.load_file(inputFilePath.c_str());
    if (!res)
        llvm::errs() << "Can't load XML file. " << res.description() << ".\n";

    traverseObjects(doc.child("program").child("objects"));

    return 0;
}
