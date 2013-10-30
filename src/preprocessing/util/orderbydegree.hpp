

/**
 * Reads preprocessed shards and creates new ones after ordering by
 * degree. Significant wasted computation here. In general, a dirty hack.
 * Used by the triangle counting example application.
 *
 * Note: for simplicity it is assumed that degree file fits into memory.
 */

#ifndef DEF_GRAPHCHI_ORDERBYDEGREE
#define DEF_GRAPH

namespace graphchi {

    struct vertex_degree {
        int deg;
        vid_t id;
        vertex_degree() {}
        vertex_degree(int deg, vid_t id) : deg(deg), id(id) {}
    };
    static bool vertex_degree_less(const vertex_degree &a, const vertex_degree &b);
    static bool vertex_degree_less(const vertex_degree &a, const vertex_degree &b) {
        return a.deg < b.deg || (a.deg == b.deg && a.id < b.id);
    }
    
     
    /**
      * Override the basic graphchi vertex class and use this for reprocessing the edges. Hack.
      */
    template <typename VT, typename ET>
    class special_sharding_vertex : public graphchi_vertex<VT, ET> {
    public:
        
        sharder<ET, ET> * sharderobj;
        vid_t * translatetable;
        special_sharding_vertex() {}
        special_sharding_vertex(sharder<ET, ET> * _sharderobj, vid_t _id, vid_t * _translatetable) : graphchi_vertex<VT, ET> (_id, NULL, NULL, 0, 0) { 
            this->sharderobj = _sharderobj;
            this->translatetable = _translatetable;
        }
        
      
        void add_inedge(vid_t src, ET * ptr, bool special_edge) {
            sharderobj->preprocessing_add_edge(translatetable[src], translatetable[this->id()]);
        }
        
         
        
        void add_outedge(vid_t dst, ET * ptr, bool special_edge) {
            // NOTE: not handled to avoid duplicates
        }
         
    };
    
    

    template <typename EdgeDataType>
    int order_by_degree(std::string &base_filename, int nshards, metrics &m) {
        /* Load degree file */
        std::string degree_filename = filename_degree_data(base_filename);
        
        degree * degrees;
        int f = open(degree_filename.c_str(), O_RDONLY);
        assert(f);
        
        size_t nverts = readfull<degree>(f, &degrees) / sizeof(degree);
        logstream(LOG_INFO) << "Starting to order by degrees, number of vertices: " << nverts << std::endl;
        
        /* Create indexed degree array */
        vertex_degree * degarray = new vertex_degree[nverts];
        for(vid_t i=0; i < nverts; i++) {
            degarray[i].id = i;
            degarray[i].deg = degrees[i].indegree + degrees[i].outdegree;
        }
        
        delete degrees;
        degrees = NULL;  
        
        logstream(LOG_INFO) << "Sorting degrees... " << std::endl;
        
        /* Sort -- TODO: use radix sort */
        quickSort(degarray, (int)nverts, vertex_degree_less);
        
        /* Create translation table */
        vid_t * translate_table = new vid_t[nverts];
        for(vid_t i=0; i<nverts; i++) {
            translate_table[degarray[i].id] = i;
        }
        delete[] degarray;
        
        logstream(LOG_INFO) << "Created translation table. " << std::endl;
        /* Write translate table */
        std::string translate_table_file = base_filename + ".vertexmap";
        int df = open(translate_table_file.c_str(), O_RDWR | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
        if (df < 0) logstream(LOG_ERROR) << "Could not write vertex map: " << translate_table_file <<
            " error: " << strerror(errno) << std::endl;
        assert(df >= 0);
        pwritea(df, translate_table, nverts * sizeof(vid_t), 0);
        close(df);
        
        logstream(LOG_INFO) << "Translation map saved into file: " << translate_table_file << std::endl;
        
        
        /* Now REPROCESS --- this is hacky! ... */
        stripedio * iomgr = new stripedio(m);
        
        /* Load intervals */
        std::vector<std::pair<vid_t, vid_t> > intervals;
        load_vertex_intervals(base_filename, nshards, intervals);
         
        std::string convertedfilename = base_filename + "_degord";
        
        /* Initialize sharder for the reprocessed graph */
        sharder<EdgeDataType, EdgeDataType> sharderobj(convertedfilename);
        sharderobj.start_preprocessing();
        
        /* Reprocess one shard a time */
        for(int p=0; p<nshards; p++) {
            logstream(LOG_INFO) << "Reprocessing shard " << p << std::endl;
            size_t nvertices = intervals[p].second - intervals[p].first + 1;
            std::vector<special_sharding_vertex<int, EdgeDataType> > vertices(nvertices, special_sharding_vertex<int, EdgeDataType>()); // preallocate
            for(size_t j=0; j<vertices.size(); j++) {
                vertices[j] = special_sharding_vertex<int, EdgeDataType>(&sharderobj, (vid_t) (intervals[p].first + j), translate_table);
                vertices[j].scheduled = true; // Otherwise memory shard will not add any edge
            }
            memory_shard<int, EdgeDataType, special_sharding_vertex<int, EdgeDataType> > memshard(iomgr, filename_shard_edata<EdgeDataType>(base_filename, p, nshards), 
                                filename_shard_adj(base_filename, p, nshards),
                                intervals[p].first, intervals[p].second, 1024 * 1024, m);
            memshard.only_adjacency = true;
            memshard.disable_parallel_loading();
            memshard.load();
            memshard.load_vertices(intervals[p].first, intervals[p].second, vertices);
        }
        
        sharderobj.end_preprocessing();
        delete iomgr;
        delete[] translate_table;

        /* Finish sharding */
        std::stringstream ss; ss << nshards;
        return sharderobj.execute_sharding(ss.str());
    }

};

#endif