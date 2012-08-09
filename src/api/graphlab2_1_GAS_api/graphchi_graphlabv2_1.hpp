
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
 * Wrapper classes for GraphLab v2.1 API.
 */

#ifndef DEF_GRAPHLAB_WRAPPERS
#define DEF_GRAPHLAB_WRAPPERS

#include "graphchi_basic_includes.hpp"

using namespace graphchi;
 

namespace graphlab {
    
    struct IS_POD_TYPE { };
    struct empty {};
    
    enum edge_dir_type {
        /**
         * \brief No edges implies that no edges are processed during the
         * corresponding gather or scatter phase, essentially skipping
         * that phase.
         */
        NO_EDGES = 0, 
        /**
         * \brief In edges implies that only whose target is the center
         * vertex are processed during gather or scatter.
         */
        IN_EDGES = 1, 
        /**
         * \brief Out edges implies that only whose source is the center
         * vertex are processed during gather or scatter.
         */
        OUT_EDGES = 2 , 
        /**
         * \brief All edges implies that all adges adjacent to a the
         * center vertex are processed on gather or scatter.  Note that
         * some neighbors may be encountered twice if there is both an in
         * and out edge to that neighbor.
         */
        ALL_EDGES = 3
    };
    
    
    typedef vid_t vertex_id_type;
    
    template<typename GraphType,
    typename GatherType, 
    typename MessageType>
    class icontext {
    public:
        // Type members ===========================================================
        
        /**
         * \brief the user graph type (typically \ref distributed_graph)
         */
        typedef GraphType graph_type;   
        
        /**
         * \brief the opaque vertex_type defined in the ivertex_program::graph_type
         * (typically distributed_graph::vertex_type)
         */
        typedef typename graph_type::vertex_type vertex_type;
        
        /**
         * \brief the global vertex identifier (see
         * graphlab::vertex_id_type).
         */
        typedef typename graph_type::vertex_id_type vertex_id_type;
        
        /**
         * The message type specified by the user-defined vertex-program.
         * (see ivertex_program::message_type)
         */
        typedef MessageType message_type;
        
        /**
         * The type returned by the gather operation.  (see
         * ivertex_program::gather_type)
         */
        typedef GatherType gather_type;
        
        /* GraphChi */
        graphchi_context * gcontext;
        
        
        
    public:        
        
        icontext(graphchi_context * gcontext) : gcontext(gcontext) {}
        
        /** \brief icontext destructor */
        virtual ~icontext() { }
        
        /**
         * \brief Get the total number of vertices in the graph.
         *
         * \return the total number of vertices in the entire graph.
         */
        virtual size_t num_vertices() const { return gcontext->nvertices; }
        
        /**
         * \brief Get the number of edges in the graph.
         *
         * Each direction counts as a separate edge.
         *
         * \return the total number of edges in the entire graph.
         */
        virtual size_t num_edges() const { assert(false); return 0; } // Not implemented yet 
        
        /**
         * \brief Get the id of this process.
         *
         * The procid is a number between 0 and 
         * \ref graphlab::icontext::num_procs
         * 
         * \warning Each process may have many threads
         *
         * @return the process of this machine.
         */
            virtual size_t procid() const { return (size_t) omp_get_thread_num(); }
        
        /**
         * \brief Returns a standard output object (like cout)
         *        which only prints once even when running distributed.
         * 
         * This returns a C++ standard output stream object
         * which maps directly to std::cout on machine with 
         * process ID 0, and to empty output streamss
         * on all other processes. Calling,
         * \code
         *   context.cout() << "Hello World!";
         * \endcode
         * will therefore only print if the code is run on machine 0.
         * This is useful in the finalize operation in aggregators.
         */
        virtual std::ostream& cout() const { return std::cout; }
        
        /**
         * \brief Returns a standard error object (like cerr)
         *        which only prints once even when running distributed.
         * 
         * This returns a C++ standard output stream object
         * which maps directly to std::cerr on machine with 
         * process ID 0, and to empty output streamss
         * on all other processes. Calling,
         * \code
         *   context.cerr() << "Hello World!";
         * \endcode
         * will therefore only print if the code is run on machine 0.
         * This is useful in the finalize operation in aggregators.
         */
        virtual std::ostream& cerr() const { return std::cerr; } 
        
        /**
         * \brief Get the number of processes in the current execution.
         *
         * This is typically the number of mpi jobs created:
         * \code
         * %> mpiexec -n 16 ./pagerank
         * \endcode
         * would imply that num_procs() returns 16.
         *
         * @return the number of processes in the current execution
         */
        virtual size_t num_procs() const { return gcontext->execthreads; }
        
        /**
         * \brief Get the elapsed time in seconds since start was called.
         * 
         * \return runtine in seconds
         */
        virtual float elapsed_seconds() const {  return gcontext->runtime(); }
        
        /**
         * \brief Return the current interation number (if supported).
         *
         * \return the current interation number if support or -1
         * otherwise.
         */
        virtual int iteration() const { return gcontext->iteration; } 
        
        /**
         * \brief Signal the engine to stop executing additional update
         * functions.
         *
         * \warning The execution engine will stop *eventually* and
         * additional update functions may be executed prior to when the
         * engine stops. For-example the synchronous engine (see \ref
         * synchronous_engine) will complete the current super-step before
         * terminating.
         */
        virtual void stop() { 
            gcontext->last_iteration = gcontext->iteration;
        } 
        
        /**
         * \brief Signal a vertex with a particular message.
         *
         * This function is an essential part of the GraphLab abstraction
         * and is used to encode iterative computation. Typically a vertex
         * program will signal neighboring vertices during the scatter
         * phase.  A vertex program may choose to signal neighbors on when
         * changes made during the previos phases break invariants or warrant
         * future computation on neighboring vertices.
         * 
         * The signal function takes two arguments. The first is mandatory
         * and specifies which vertex to signal.  The second argument is
         * optional and is used to send a message.  If no message is
         * provided then the default message is used.
         *
         * \param vertex [in] The vertex to send the message to
         * \param message [in] The message to send, defaults to message_type(). 
         */
        virtual void signal(const vertex_type& vertex, 
                            const message_type& message = message_type()) { 
            gcontext->scheduler->add_task(vertex.id());
        }
        
        /**
         * \brief Send a message to a vertex ID.
         *
         * \warning This function will be slow since the current machine
         * do not know the location of the vertex ID.  If possible use the
         * the icontext::signal call instead.
         *
         * \param gvid [in] the vertex id of the vertex to signal
         * \param message [in] the message to send to that vertex, 
         * defaults to message_type().
         */
        virtual void signal_vid(vertex_id_type gvid, 
                                const message_type& message = message_type()) {
            gcontext->scheduler->add_task(gvid);
        }
        
        /**
         * \brief Post a change to the cached sum for the vertex
         * 
         * Often a vertex program will be signaled due to a change in one
         * or a few of its neighbors.  However the gather operation will
         * be rerun on all neighbors potentially producing the same value
         * as previous invocations and wasting computation time.  To
         * address this some engines support caching (see \ref
         * gather_caching for details) of the gather phase.
         *
         * When caching is enabled the engines save a copy of the previous
         * gather for each vertex.  On subsequent calls to gather if their
         * is a cached gather then the gather phase is skipped and the
         * cached value is passed to the ivertex_program::apply function.
         * Therefore it is the responsibility of the vertex program to
         * update the cache values for neighboring vertices. This is
         * accomplished by using the icontext::post_delta function.
         * Posted deltas are atomically added to the cache.
         *
         * \param vertex [in] the vertex whose cache we want to update
         * \param delta [in] the change that we want to *add* to the
         * current cache.
         *
         */
        virtual void post_delta(const vertex_type& vertex, 
                                const gather_type& delta) { 
            assert(false); // Not implemented
        } 
        
        /**
         * \brief Invalidate the cached gather on the vertex.
         *
         * When caching is enabled clear_gather_cache clears the cache
         * entry forcing a complete invocation of the subsequent gather.
         *
         * \param vertex [in] the vertex whose cache to clear.
         */
        virtual void clear_gather_cache(const vertex_type& vertex) {
            assert(false); // Not implemented
        } 
        
    }; // end of icontext
    

    /* Forward declaratinos */
    template <typename GLVertexDataType, typename EdgeDataType>
    struct GraphLabVertexWrapper;
    
    template <typename GLVertexDataType, typename EdgeDataType>
    struct GraphLabEdgeWrapper;
    
    /* Fake distributed graph type (this is often hard-coded
     in GraphLab vertex programs. */
    template <typename vertex_data, typename edge_data>
    struct distributed_graph {
        typedef vertex_data vertex_data_type;
        typedef edge_data edge_data_type;
        typedef GraphLabVertexWrapper<vertex_data_type, edge_data_type> vertex_type;
        typedef GraphLabEdgeWrapper<vertex_data_type, edge_data_type> edge_type;
        typedef graphchi::vid_t vertex_id_type;
    };
    
    
    /* GraphChi's version of the ivertex_program */
    template<typename Graph,
    typename GatherType, typename MessageType = bool> 
    struct ivertex_program {
        
        /* Type definitions */
        typedef typename Graph::vertex_data_type vertex_data_type;
        typedef typename Graph::edge_data_type edge_data_type;
        typedef GatherType gather_type;
        typedef MessageType message_type;
        
        typedef Graph graph_type;
        typedef typename graphchi::vid_t vertex_id_type;
        typedef GraphLabVertexWrapper<vertex_data_type, edge_data_type> vertex_type;
        typedef GraphLabEdgeWrapper<vertex_data_type, edge_data_type> edge_type;
        typedef icontext<graph_type, gather_type, message_type> icontext_type;
        
        typedef graphlab::edge_dir_type edge_dir_type;
        
        virtual void init(icontext_type& context,
                          const vertex_type& vertex, 
                          const message_type& msg) { /** NOP */ }
        
        /**
         * Returns the set of edges on which to run the gather function.
         * The default edge direction is the in edges.
         */
        virtual edge_dir_type gather_edges(icontext_type& context,
                                           const vertex_type& vertex) const { 
            return IN_EDGES; 
        }
        
        /**
         * Gather is called on all gather_edges() in parallel and returns
         * the gather_type which are added to compute the final output of
         * the gather.
         */
        virtual gather_type gather(icontext_type& context, 
                                   const vertex_type& vertex, 
                                   edge_type& edge) const {
            logstream(LOG_FATAL) << "Gather not implemented!" << std::endl;
            return gather_type();
        };
        
        /**
         * The apply function is called once the gather has completed and
         * must be implemented by all vertex programs. 
         */
        virtual void apply(icontext_type& context, 
                           vertex_type& vertex, 
                           const gather_type& total) = 0;
        
        /**
         * Returns the set of edges on which to run the scatter function.
         * The default edge direction is the out edges.
         */
        virtual edge_dir_type scatter_edges(icontext_type& context,
                                            const vertex_type& vertex) const { 
            return OUT_EDGES; 
        }
        
        /**
         * Scatter is called on all scatter_edges() in parallel after the
         * apply function has completed.  The scatter function can post
         * deltas.
         */
        virtual void scatter(icontext_type& context, const vertex_type& vertex, 
                             edge_type& edge) const { 
            logstream(LOG_FATAL) << "Scatter not implemented!" << std::endl;
        };
    };
    
    template <typename GLVertexDataType, typename EdgeDataType>
    struct GraphLabVertexWrapper {
        typedef graphchi_vertex<bool, EdgeDataType> VertexType; // Confusing!
        typedef GLVertexDataType vertex_data_type;
        typedef GraphLabVertexWrapper<GLVertexDataType, EdgeDataType> vertex_type;
        
        graphchi::vid_t vertexId;
        VertexType * vertex;
        std::vector<GLVertexDataType> * vertexArray;
        
        GraphLabVertexWrapper(graphchi::vid_t vertexId, VertexType * vertex,
                                    std::vector<GLVertexDataType> * vertexArray): vertexId(vertexId), 
                                        vertex(vertex), vertexArray(vertexArray) { }
        
        bool operator==(vertex_type& other) const {
            return vertexId == other.vertexId;
        }
        
        /// \brief Returns a constant reference to the data on the vertex
        const vertex_data_type& data() const {
            return (*vertexArray)[vertexId];
        }
        
        /// \brief Returns a mutable reference to the data on the vertex
        vertex_data_type& data() {
            return (*vertexArray)[vertexId];
        }
        
        /// \brief Returns the number of in edges of the vertex
        size_t num_in_edges() const {
            if (vertex == NULL) {
                logstream(LOG_ERROR) << "GraphChi does not support asking neighbor vertices in/out degrees." << std::endl;
                return 0;
            }
            return vertex->num_edges();
        }
        
        /// \brief Returns the number of out edges of the vertex
        size_t num_out_edges() const {
            if (vertex == NULL) {
                logstream(LOG_ERROR) << "GraphChi does not support asking neighbor vertices in/out degrees." << std::endl;
                return 0;
            }
            return vertex->num_outedges();
        }
        
        /// \brief Returns the vertex ID of the vertex       
        graphchi::vid_t id() const {
            return vertexId;
        }
        
        /** 
         *  \brief Returns the local ID of the vertex
         */
        graphchi::vid_t local_id() const {
            return vertexId;
        }
        
    };
    
    
    template <typename GLVertexDataType, typename EdgeDataType>
    struct GraphLabEdgeWrapper {
        typedef graphchi_vertex<bool, EdgeDataType> VertexType;
        typedef GLVertexDataType vertex_data_type;
        typedef EdgeDataType edge_data_type;
        typedef GraphLabVertexWrapper<GLVertexDataType, EdgeDataType> vertex_type;

        graphchi_edge<EdgeDataType> * edge;
        VertexType * vertex;
        std::vector<GLVertexDataType> * vertexArray;
        bool is_inedge;
        
        GraphLabEdgeWrapper(graphchi_edge<EdgeDataType> * edge, VertexType * vertex,
                         std::vector<GLVertexDataType> * vertexArray, bool is_inedge): 
        edge(edge), vertex(vertex), vertexArray(vertexArray), is_inedge(is_inedge) { }
        
             
    public:
        
        /**
         * \brief Returns the source vertex of the edge. 
         * This function returns a vertex_object by value and as a
         * consequence it is possible to use the resulting vertex object
         * to access and *modify* the associated vertex data.
         *
         * Modification of vertex data obtained through an edge object
         * is *usually not safe* and can lead to data corruption.
         *
         * \return The vertex object representing the source vertex.
         */
        vertex_type source() const { 
            if (is_inedge) {
                return GraphLabVertexWrapper<GLVertexDataType, EdgeDataType>(vertex->id(), vertex, vertexArray); 
            } else {
                return GraphLabVertexWrapper<GLVertexDataType, EdgeDataType>(edge->vertex_id(), NULL, vertexArray); 
            }
        }
        
        /**
         * \brief Returns the target vertex of the edge. 
         *
         * This function returns a vertex_object by value and as a
         * consequence it is possible to use the resulting vertex object
         * to access and *modify* the associated vertex data.
         *
         * Modification of vertex data obtained through an edge object
         * is *usually not safe* and can lead to data corruption.
         *
         * \return The vertex object representing the target vertex.
         */
        vertex_type target() const { 
            if (!is_inedge) {
                return GraphLabVertexWrapper<GLVertexDataType, EdgeDataType>(vertex->id(), vertex, vertexArray); 
            } else {
                return GraphLabVertexWrapper<GLVertexDataType, EdgeDataType>(edge->vertex_id(), NULL, vertexArray); 
            }
        }
        
        /**
         * \brief Returns a constant reference to the data on the edge 
         */
        const edge_data_type& data() const { return const_cast<edge_data_type&>(*edge->data_ptr); }
        
        /**
         * \brief Returns a mutable reference to the data on the edge 
         */
        edge_data_type& data() { return *(edge->data_ptr); }
        
    }; // end of edge_type
    
    
    template <class GraphLabVertexProgram>
    struct GraphLabWrapper : public GraphChiProgram<bool, typename GraphLabVertexProgram::edge_data_type> {
        typedef bool VertexDataType;  /* Temporary hack: as the vertices are stored in memory, no need to store on disk. */
        typedef typename GraphLabVertexProgram::vertex_data_type GLVertexDataType;
        typedef typename GraphLabVertexProgram::edge_data_type EdgeDataType;
        typedef typename GraphLabVertexProgram::gather_type gather_type;
        typedef typename GraphLabVertexProgram::graph_type graph_type;
        typedef typename GraphLabVertexProgram::message_type message_type;
        
        std::vector<GLVertexDataType> * vertexInmemoryArray;
     
        GraphLabWrapper() {
            vertexInmemoryArray = new std::vector<GLVertexDataType>();
        }
        
        /**
         * Called before an iteration starts.
         */
        virtual void before_iteration(int iteration, graphchi_context &gcontext) {
            if (gcontext.iteration == 0) {
                logstream(LOG_INFO) << "Initialize vertices in memory." << std::endl;
                vertexInmemoryArray->resize(gcontext.nvertices);
            }
        }
        
        /**
         * Called after an iteration has finished.
         */
        virtual void after_iteration(int iteration, graphchi_context &gcontext) {
        }
        
        /**
         * Called before an execution interval is started.
         */
        virtual void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
        }
        
        /**
         * Called after an execution interval has finished.
         */
        virtual void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
        }
        
        /**
         * Update function.
         */
        void update(graphchi_vertex<bool, EdgeDataType> &vertex, graphchi_context &gcontext) {
            graphlab::icontext<graph_type, gather_type, message_type> glcontext(&gcontext);
            
            /* Create the vertex program */
            GraphLabVertexWrapper<GLVertexDataType, EdgeDataType> wrapperVertex(vertex.id(), &vertex, vertexInmemoryArray);
            GraphLabVertexProgram glVertexProgram;
            
            /* Init */
            glVertexProgram.init(glcontext, wrapperVertex, typename GraphLabVertexProgram::message_type());
            const GraphLabVertexProgram& const_vprog = glVertexProgram;
            
            /* Gather */
            edge_dir_type gather_direction = const_vprog.gather_edges(glcontext, wrapperVertex);
            gather_type sum;
            
            int gathered = 0;
            switch (gather_direction) {
                case ALL_EDGES:
                case IN_EDGES:
                    for(int i=0; i < vertex.num_inedges(); i++) {
                        GraphLabEdgeWrapper<GLVertexDataType, EdgeDataType> edgeWrapper(vertex.inedge(i), &vertex, vertexInmemoryArray, true);
                        if (gathered > 0) sum += const_vprog.gather(glcontext, wrapperVertex, edgeWrapper);
                        else sum = const_vprog.gather(glcontext, wrapperVertex, edgeWrapper);
                        gathered++;
                    }
                    if (gather_direction != ALL_EDGES)
                        break;
                case OUT_EDGES:
                    for(int i=0; i < vertex.num_outedges(); i++) {
                        GraphLabEdgeWrapper<GLVertexDataType, EdgeDataType> edgeWrapper(vertex.outedge(i), &vertex, vertexInmemoryArray, false);
                        if (gathered > 0) sum += const_vprog.gather(glcontext, wrapperVertex, edgeWrapper);
                        else sum = const_vprog.gather(glcontext, wrapperVertex, edgeWrapper);
                        gathered++;
                    }
                    break;
                case NO_EDGES:
                    break;
                default:
                    assert(false); // Huh?
            }
            
            
            /* Apply */
            glVertexProgram.apply(glcontext, wrapperVertex, sum);
            
            /* Scatter */
            edge_dir_type scatter_direction = const_vprog.scatter_edges(glcontext, wrapperVertex);
            
            switch(scatter_direction) {
                case ALL_EDGES:
                case IN_EDGES:
                    for(int i=0; i < vertex.num_inedges(); i++) {
                        GraphLabEdgeWrapper<GLVertexDataType, EdgeDataType> edgeWrapper(vertex.inedge(i), &vertex, vertexInmemoryArray, true);
                        const_vprog.scatter(glcontext, wrapperVertex, edgeWrapper);
                    }    
                    if (scatter_direction != ALL_EDGES)
                        break;
                case OUT_EDGES:
                    for(int i=0; i < vertex.num_outedges(); i++) {
                        GraphLabEdgeWrapper<GLVertexDataType, EdgeDataType> edgeWrapper(vertex.outedge(i), &vertex, vertexInmemoryArray, false);
                        const_vprog.scatter(glcontext, wrapperVertex, edgeWrapper);
                    }    
                    break;
                case NO_EDGES:
                    break;
                default:
                    assert(false); // Huh?
            }
            
            /* Done! */
        }
        
        
    }; // End GraphLabWrapper
    
    template <typename GraphLabVertexProgram, typename ReductionType,
    typename EdgeMapType,
    typename FinalizerType>
    struct GraphLabEdgeAggregatorWrapper : public GraphChiProgram<bool, typename GraphLabVertexProgram::edge_data_type> {
        typedef bool VertexDataType;  /* Temporary hack: as the vertices are stored in memory, no need to store on disk. */
        typedef typename GraphLabVertexProgram::vertex_data_type GLVertexDataType;
        typedef typename GraphLabVertexProgram::edge_data_type EdgeDataType;
        typedef typename GraphLabVertexProgram::edge_type edge_type;
        typedef typename GraphLabVertexProgram::gather_type gather_type;
        typedef typename GraphLabVertexProgram::graph_type graph_type;
        typedef typename GraphLabVertexProgram::message_type message_type;
        
        mutex m;
        std::vector<ReductionType> localaggr;
        ReductionType aggr;
        std::vector<GLVertexDataType> * vertexInmemoryArray;
        EdgeMapType map_function;
        FinalizerType finalize_function;
        
        GraphLabEdgeAggregatorWrapper(EdgeMapType map_function,
                                      FinalizerType finalize_function, 
                                      std::vector<typename GraphLabVertexProgram::vertex_data_type> * vertices) : map_function(map_function),
                                            finalize_function(finalize_function) {
            vertexInmemoryArray = vertices;
        }

        /**
         * Called before an iteration starts.
         */
        virtual void before_iteration(int iteration, graphchi_context &gcontext) {
            aggr = ReductionType();
            localaggr.resize(gcontext.execthreads);
        }
        
        /**
         * Called after an iteration has finished.
         */
        virtual void after_iteration(int iteration, graphchi_context &gcontext) {
            logstream(LOG_INFO) << "Going to run edge-aggregator finalize." << std::endl;
            
            for(int i=0; i < (int)localaggr.size(); i++) {
                aggr += localaggr[i];
            }
            
            graphlab::icontext<graph_type, gather_type, message_type> glcontext(&gcontext);
            finalize_function(glcontext, aggr);
        }
        
        /**
         * Called before an execution interval is started.
         */
        virtual void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
        }
        
        /**
         * Called after an execution interval has finished.
         */
        virtual void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
        }
        
        /**
         * Update function.
         */
        void update(graphchi_vertex<bool, EdgeDataType> &vertex, graphchi_context &gcontext) {
            graphlab::icontext<graph_type, gather_type, message_type> glcontext(&gcontext);
            ReductionType a;
            for(int i=0; i < vertex.num_edges(); i++) {
                const GraphLabEdgeWrapper<GLVertexDataType, EdgeDataType> edgeWrapper(vertex.edge(i), &vertex, vertexInmemoryArray, true);
                ReductionType mapped = map_function(glcontext, edgeWrapper);
                a += mapped;
            }    
            localaggr[omp_get_thread_num()] += a;
        }
    }; // End edge-aggregator wrapper
        
    
         
    /**
      * Just definitions, we do not actually 
        support them.
      */
        namespace messages {
            
            /**
             * The priority of two messages is the sum
             */
            struct sum_priority : public graphlab::IS_POD_TYPE {
                double value;
                sum_priority(const double value = 0) : value(value) { }
                double priority() const { return value; }
                sum_priority& operator+=(const sum_priority& other) {
                    value += other.value;
                    return *this;
                }
            }; // end of sum_priority message
            
            /**
             * The priority of two messages is the max
             */
            struct max_priority : public graphlab::IS_POD_TYPE {
                double value;
                max_priority(const double value = 0) : value(value) { }
                double priority() const { return value; }
                max_priority& operator+=(const max_priority& other) {
                    value = std::max(value, other.value);
                    return *this;
                }
            }; // end of max_priority message
            
            
        }; // end of messages namespace
        
        
     
    
}; // End namespace graphlab

template <typename GraphLabVertexProgram>
    std::vector<typename GraphLabVertexProgram::vertex_data_type> *
            run_graphlab_vertexprogram(std::string base_filename, int nshards, int niters, bool scheduler, metrics & _m,
                                    bool modifies_inedges=true, bool modifies_outedges=true) {
    typedef graphlab::GraphLabWrapper<GraphLabVertexProgram> GLWrapper;
    GLWrapper wrapperProgram;
    graphchi_engine<bool, typename GLWrapper::EdgeDataType> engine(base_filename, nshards, scheduler, _m); 
    engine.set_modifies_inedges(modifies_inedges);
    engine.set_modifies_outedges(modifies_outedges);
    engine.run(wrapperProgram, niters);
               return wrapperProgram.vertexInmemoryArray;
}

template <typename GraphLabVertexProgram, typename ReductionType,
    typename EdgeMapType,
    typename FinalizerType>
ReductionType run_graphlab_edge_aggregator(std::string base_filename, int nshards,
                                  EdgeMapType map_function,
                                  FinalizerType finalize_function, std::vector<typename GraphLabVertexProgram::vertex_data_type> * vertices, metrics & _m) {
    typedef graphlab::GraphLabEdgeAggregatorWrapper<GraphLabVertexProgram, ReductionType, EdgeMapType, FinalizerType> GLEdgeAggrWrapper;
    logstream(LOG_INFO) << "Starting edge aggregator." << std::endl;
    GLEdgeAggrWrapper glAggregator(map_function, finalize_function, vertices);
    graphchi_engine<bool, typename GLEdgeAggrWrapper::EdgeDataType> engine(base_filename, nshards, true, _m); 
    engine.set_modifies_inedges(false);
    engine.set_modifies_outedges(false);
    engine.run(glAggregator, 1);
    return glAggregator.aggr;
}


#endif

