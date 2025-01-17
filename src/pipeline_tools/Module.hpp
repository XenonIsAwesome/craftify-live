#ifndef MODULE_HPP
#define MODULE_HPP

#include "UntypedModule.h"

namespace pipeline_tools {
    template<typename InT, typename OutT>
    class Module: public UntypedModule {
        public:
            Module() = default;

            std::shared_ptr<void> untyped_process(std::shared_ptr<void> input) override {
                /// Casting input
                std::shared_ptr<InT> typed_input = std::static_pointer_cast<InT>(input);

                /// Running the process method
                std::shared_ptr<OutT> typed_output = process(typed_input);
                
                /// Casting output
                return std::static_pointer_cast<void>(typed_output);
            };

            virtual std::shared_ptr<OutT> process(std::shared_ptr<InT> input) = 0;

    };
}
#endif // MODULE_HPP
