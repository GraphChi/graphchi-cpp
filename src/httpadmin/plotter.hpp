
/**
 * @file
 * @author  Aapo Kyrola <akyrola@cs.cmu.edu>
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Aapo Kyrola, Guy Blelloch, Carlos Guestrin / Carnegie Mellon University]
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 
 *
 * @section DESCRIPTION
 *
 * Code that generates periodically engine status and performance
 * related plots using python's matplotlib.
 *
 * Relatively hacky code.
 */
#ifndef DEF_GRAPHCHI_GNUPLOTTER
#define DEF_GRAPHCHI_GNUPLOTTER

#include <cstdio>
#include <stdlib.h>
#include <string>

#include "graphchi_basic_includes.hpp"

namespace graphchi {
    
    static size_t initial_edges = 0;

    
    static std::string plotdirectory();
    static std::string plotdirectory() {
        return "conf/adminhtml/plots/";}

    static  void init_plot(std::string plotname);
    static void init_plot(std::string plotname) {
        std::string dataname = plotdirectory() + plotname + ".dat";
        FILE * df = fopen(dataname.c_str(), "w");
        fclose(df);
        std::cout << "---------- Initialized ------------" << std::endl;
    }
    
    template <typename ENGINE>
    void addval(ENGINE * engine, std::string plotname, double val) {
        graphchi_context &context = engine->get_context();
        std::string dataname = plotdirectory() + plotname + ".dat";
        FILE * df = fopen(dataname.c_str(), "a");
        assert(df != NULL);
        fprintf(df, "%lf %lf\n", context.runtime(), val);
        fclose(df);

    }
    
    static void drawplot(std::string plotname, size_t lookback_secs);
    static void drawplot(std::string plotname, size_t lookback_secs) {
        std::string plotfile = plotdirectory() + plotname + ".dat";
        
        std::stringstream ss;
        ss << "python ";
        ss <<  plotdirectory() + "plotter.py " + plotfile + " lastsecs ";
        ss << lookback_secs;
        
        std::string cmd = ss.str();
        logstream(LOG_DEBUG) << "Executing: " << cmd << std::endl;
        system(cmd.c_str());
    }
    
    template <typename ENGINE> 
    static void init_plots(ENGINE * engine) {
        init_plot("edges");
        init_plot("bufedges");
        init_plot("updates");
        init_plot("ingests");
        init_plot("deltas");

        initial_edges = engine->num_edges_safe();
    }
    
    static double last_update_time = 0;
    static size_t last_edges = 0;
    static size_t last_updates = 0;
    
 
    template <typename ENGINE> 
    void update_plotdata(ENGINE * engine) {
        addval(engine, "edges", (double)engine->num_edges_safe());
        addval(engine, "bufedges", (double)engine->num_buffered_edges());
        double rt = engine->get_context().runtime() - last_update_time;

        if (last_update_time > 0) {
            addval(engine, "ingests", (engine->num_edges_safe() - last_edges) / rt);
            addval(engine, "updates", (engine->num_updates() - last_updates) / rt);
            addval(engine, "deltas", engine->get_context().last_deltasum);
        }
        if (last_update_time == 0 || rt >= 120.0) {
            last_edges = engine->num_edges_safe();
            last_update_time = engine->get_context().runtime();
            last_updates = engine->num_updates();
        }
    }
    
    static void drawplots();
    static void drawplots() {
        drawplot("edges", 1800);
        drawplot("bufedges", 1800);
        drawplot("updates", 300);
        drawplot("ingests", 500);
        drawplot("deltas", 7200);

    }

}

#endif
