#ifndef PIPELINE_H
#define PIPELINE_H

#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include "UntypedModule.h"

namespace pipeline_tools {
    class Pipeline {
        public:
            Pipeline(std::vector<std::shared_ptr<UntypedModule>> modules);

            void start();

            void stop();

            std::atomic<bool>& get_running_ref();

    private:
        std::atomic<bool> running_ref;
        std::vector<std::shared_ptr<UntypedModule>> pipeline_modules;

        std::thread worker;

    };
}
#endif // PIPELINE_H
