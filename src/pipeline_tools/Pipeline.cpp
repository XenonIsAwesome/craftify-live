#include "Pipeline.h"

namespace pipeline_tools {

    Pipeline::Pipeline(std::vector<std::shared_ptr<UntypedModule>> modules): 
        pipeline_modules(std::move(modules)), running_ref(true) {}

    void Pipeline::start() {
        this->worker = std::thread([&](){
            std::shared_ptr<void> work_item;

            while (this->running_ref.load()) {
                std::shared_ptr<void> work_item = nullptr;
                for (std::shared_ptr<UntypedModule> mod: this->pipeline_modules) {
                    work_item = mod->untyped_process(work_item); 
                    if (work_item.get() == nullptr) {
                        break;
                    }
                }
            }
        });
    }

    void Pipeline::stop() {
        this->running_ref.store(false);
        worker.join();
    }

    std::atomic<bool>& Pipeline::get_running_ref() {
        return this->running_ref;
    }

}
