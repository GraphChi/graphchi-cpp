// Adapted from http://jsfiddle.net/7HZcR/3/

var w = 960,
h = 800,
input = "graph1000.csv";

// Read file and operate
d3.csv(input, function(links) {

// Build nodes object
var nodes = {};
//links = links.slice(1,80);
for (var i = 0; i < links.length; i++){
	link = links[i];
	link.source = nodes[link.source] || (nodes[link.source] = {name: link.source});
	link.target = nodes[link.target] || (nodes[link.target] = {name: link.target});
}


// Draw it
var force = d3.layout.force()
.nodes(d3.values(nodes))
.links(links)
.size([w, h])
.charge(-200)
.gravity(.5)
.on("tick", tick)
.start();

var svg = d3.select("#large_graph").append("svg:svg")
.attr("width", w)
.attr("height", h);


var path = svg.append("svg:g").selectAll("path")
.data(force.links())
.enter().append("svg:path")
.attr("class", function(d) { return "link"; });

var circle = svg.append("svg:g").selectAll("circle")
.data(force.nodes())
.enter().append("svg:circle").style("fill", function(d) { 
	if (d.name <10000) return "red";
	else return "steelblue";
})
.attr("r", function(d) {
	return d.weight;
})
.on("mouseover", function(){d3.select(this).style("fill", "aliceblue");})
.on("mouseout", function(){d3.select(this).style("fill", "");})
.call(force.drag);


var text = svg.append("svg:g").selectAll("g")
.data(force.nodes())
.enter().append("svg:g");

// A copy of the text with a thick white stroke for legibility.
text.append("svg:text")
.attr("x", 8)
.attr("y", ".31em")
.attr("class", "shadow")
.text(function(d) { return d.name; });

text.append("svg:text")
.attr("x", 8)
.attr("y", ".31em")
.text(function(d) { return d.name; });

function tick() {
	path.attr("d", function(d) {
		return "M" + d.source.x + "," + d.source.y + "A0,0 0 0,1 " + d.target.x + "," + d.target.y;
	});

	circle.attr("transform", function(d) {
		return "translate(" + d.x + "," + d.y + ")";
	});

	text.attr("transform", function(d) {
		return "translate(" + d.x + "," + d.y + ")";
	});
}

});


