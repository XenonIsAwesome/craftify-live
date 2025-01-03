#ifndef DEVICEEXECUTER_H
#define DEVICEEXECUTER_H

#include <memory>

namespace craftify {
    namespace executers {
        template<typename InT, typename OutT>
        class DeviceExecuter{
        public:
            virtual std::shared_ptr<OutT> execute(std::shared_ptr <InT> input) = 0;
        };
    }; // namespace executers
}; // namespace craftify

#endif //DEVICEEXECUTER_H
