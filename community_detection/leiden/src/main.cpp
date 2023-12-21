#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

typedef unsigned int u32;

struct Graph {
	float* weights;
	u32* src_nodes;
	u32* dst_nodes;
	u32 num_nodes;
	u32 num_edges;
};



u32 local_moving_phase(
        Graph& graph,
        std::vector<std::vector<u32>>& sorted_edgelist,
        std::vector<std::vector<float>>& sorted_edgeweights,
        std::vector<float>& node_strengths,
        std::vector<int>& assignments,
        float total_edge_weight,
        float resolution
    ) {
    float inv_double_total_edge_weight = resolution / (2.0f * total_edge_weight);

	u32 num_swaps = 0;

	std::vector<u32> node_order(graph.num_nodes);
	for (u32 idx = 0; idx < graph.num_nodes; ++idx) {
		node_order[idx] = idx;
	}
	std::shuffle(node_order.begin(), node_order.end(), std::mt19937(std::random_device()()));

    // Iterate over each node to find the best community
    // for (u32 idx = 0; idx < graph.num_nodes; ++idx) {
    for (u32 idx : node_order) {
        int   original_community   = assignments[idx];

        float best_modularity_gain = 0.0f;
        int   best_community 	   = original_community;

        // Temporarily remove the node from its original community
        float node_strength_removed = node_strengths[idx];

        node_strengths[original_community] -= node_strength_removed;

		float ss_2w = node_strength_removed * inv_double_total_edge_weight;

        // Check the community of each neighbor
        for (u32 jdx = 0; jdx < sorted_edgelist[idx].size(); ++jdx) {
            int neighbor_community = assignments[sorted_edgelist[idx][jdx]];

            // Skip if the neighbor is in the same community as the original
            if (neighbor_community == original_community) {
                continue;
            }

            // Calculate the change in modularity if this node joins the neighbor's community
            // float delta_modularity = sorted_edgeweights[idx][jdx] - ss_2w * node_strengths[neighbor_community];
            float delta_modularity = sorted_edgeweights[idx][jdx] - node_strengths[neighbor_community];
			// std::cout << delta_modularity << "  " << ss_2w << std::endl;

            if (delta_modularity > best_modularity_gain) {
                best_modularity_gain = delta_modularity;
                best_community = neighbor_community;
            }
        }

        // Assign the node to the community with the best modularity gain
        assignments[idx] = best_community;

        // Update node strengths for the new and old communities
        if (original_community != best_community) {
            node_strengths[best_community] += node_strength_removed;
			num_swaps++;
        }
		else {
            // Revert the node strength change if the community did not change
            node_strengths[original_community] += node_strength_removed;
        }
    }
	return num_swaps;
}


std::vector<int> leiden_community_detection(
		Graph& graph,
		float resolution = 1.0,
		u32 num_iterations = 10
		) {
	/**************************************
	 ** Leiden community detection algorithm **
	 Perform community detection on a graph using the Leiden algorithm.
	 Return community assignments for each node.
	 **************************************/
	std::vector<int> assignments;
	assignments.resize(graph.num_nodes);
	for (u32 idx = 0; idx < graph.num_nodes; ++idx) {
		assignments[idx] = idx;
	}

	float total_edge_weight = 0.0f;
	#pragma omp parallel for reduction(+:total_edge_weight)
	for (u32 idx = 0; idx < graph.num_edges; ++idx) {
		total_edge_weight += graph.weights[idx];
	}

	// Get sorted edgelist.
	// For each node in src_nodes, find all links to dst_nodes.
	std::vector<std::vector<u32>> sorted_edgelist(graph.num_nodes);
	std::vector<std::vector<float>> sorted_edgeweights(graph.num_nodes);
	std::vector<float> node_strengths(graph.num_nodes);
	for (u32 idx = 0; idx < graph.num_edges; ++idx) {
		sorted_edgelist[graph.src_nodes[idx]].push_back(graph.dst_nodes[idx]);
		sorted_edgeweights[graph.src_nodes[idx]].push_back(graph.weights[idx]);
		node_strengths[graph.src_nodes[idx]] += graph.weights[idx];
		node_strengths[graph.dst_nodes[idx]] += graph.weights[idx];
	}

		
	u32 num_swaps = 0;
	for (u32 iteration = 0; iteration < num_iterations; ++iteration) {
		num_swaps = local_moving_phase(
				graph,
				sorted_edgelist,
				sorted_edgeweights,
				node_strengths,
				assignments,
				total_edge_weight,
				resolution
				);
		std::cout << "Iteration: " << iteration + 1 << "    Number of Swaps: " << num_swaps << std::endl;
		if (num_swaps == 0) {
			break;
		}
	}

	return assignments;
}




int main() {
	// Generate random sparse graph
	u32 num_nodes = 10000;
	u32 num_edges = 100000;
	std::vector<u32> src_nodes(num_edges);
	std::vector<u32> dst_nodes(num_edges);
	std::vector<float> weights(num_edges);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<u32> dis(0, num_nodes - 1);
	std::uniform_real_distribution<float> dis2(0.0, 1.0);

	for (u32 idx = 0; idx < num_edges; ++idx) {
		src_nodes[idx] = dis(gen);
		dst_nodes[idx] = dis(gen);
		weights[idx]   = dis2(gen);
	}

	Graph graph = {
		weights.data(),
		src_nodes.data(),
		dst_nodes.data(),
		num_nodes,
		num_edges
	};

	std::vector<int> assignments = leiden_community_detection(graph, 1.0, 1000000);
	printf("\n");

	// for (u32 idx = 0; idx < assignments.size(); ++idx) {
	for (u32 idx = 0; idx < 25; ++idx) {
		const char* spaces = idx < 10 ? "    " : (idx < 100 ? "   " : (idx < 1000 ? "  " : " "));
		std::cout << "Node: " << idx << spaces << "Assignment: " << assignments[idx] << std::endl;
	}

	return 0;
}
