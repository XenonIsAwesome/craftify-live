#ifndef UNTYPED_MODULE_H
#define UNTYPED_MODULE_H

// #include <typeinfo>
#include <memory>

namespace pipeline_tools {
    class UntypedModule {
    public:
        UntypedModule() = default;

        virtual std::shared_ptr<void> untyped_process(std::shared_ptr<void> input) = 0;

    };
}

#endif // UNTYPED_MODULE_H
