#ifndef DEVICEEXECUTER_H
#define DEVICEEXECUTER_H

#include <memory>
#include <optional>

namespace craftify::executers {
    template<typename InT, typename OutT>
    class DeviceExecuter{
    public:
        virtual std::optional<std::shared_ptr<OutT>> execute(std::shared_ptr <InT> input) = 0;
    };
}
#endif //DEVICEEXECUTER_H
