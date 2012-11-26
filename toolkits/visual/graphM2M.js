// Adapted from http://jsfiddle.net/7HZcR/3/

var w = 500,
    h = 500;

$.get('./file.txt', function(response){

    var data = [];
    var text_arr = response.split("\n");
    for(var i = 0; i < text_arr.length; i++) {
    if(text_arr[i].trim().length > 0) { 
    data.push(text_arr[i]);
    }
    }

    d3.select("#order")
    .selectAll("option")
    .data(data)
    .enter().append("option")
    .text(String);

    draw_graph(data[0]);		
    check_iterations('graph1500.1.csv.txt');
  
   function check_iterations(filename) {

    $.get(filename, function(response){
        var settings = parse_settings(response);
        $('#graph_table').empty();
        for (var ind=0; ind < settings.length-1; ind = ind+2) {
        $("#graph_table").append("<tr><td>" + settings[ind] + "</td><td>" + settings[ind+1] + "</td></tr>");
        }
        });
  } //end of check_iterations

  function parse_settings(text) {
    var text_arr = text.split(/[\[\]]/);
    var settings_arr = [];
    for(var i = 0; i < text_arr.length; i++) {
      if(text_arr[i].trim().length > 0) { 
        settings_arr.push(text_arr[i]);
      }
    }
    return settings_arr;
  }


});



function draw_graph(filename){

  // Read file and operate
  d3.csv(filename, function(links) {

      // Build nodes object
      var nodes = {};
      //links = links.slice(1,80);
      for (var i = 0; i < links.length; i++){
      link = links[i];
      link.source = nodes[link.source] || (nodes[link.source] = {name: link.source});
      link.target = nodes[link.target] || (nodes[link.target] = {name: link.target});
      }

      d3.layout.force().stop();
      d3.layout.force().nodes([]);
      d3.layout.force().links([]);
      d3.select("#thesvg").remove();

      // Draw it
      var force = d3.layout.force()
      .nodes(d3.values(nodes))
      .links(links)
      .size([w, h])
      .charge(-500)
      .gravity(.5)
      .on("tick", tick)
      .start();

      var svg = d3.select("#large_graph").append("svg:svg")
        .attr("width", w)
        .attr("height", h)
        .attr("id", "thesvg");

      var path = svg.append("svg:g").selectAll("path")
        .data(force.links())
        .enter().append("svg:path")
        .attr("class", function(d) { return "link"; });

      var circle = svg.append("svg:g").selectAll("circle")
        .data(force.nodes())
        .enter().append("svg:circle").style("fill", function(d) { 
            if (d.name <10000) return "red";
            else if (d.name > 150000) return "green"; else return "steelblue";
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

  }); //end of draw_graph

//input = "graph1000.csv";
  function check_iterations(filename) {

    $.get(filename, function(response){
        var settings = parse_settings(response);
        $('#graph_table').empty();
        for (var ind=0; ind < settings.length-1; ind = ind+2) {
        $("#graph_table").append("<tr><td>" + settings[ind] + "</td><td>" + settings[ind+1] + "</td></tr>");
        }
        });
  } //end of check_iterations

  function parse_settings(text) {
    var text_arr = text.split(/[\[\]]/);
    var settings_arr = [];
    for(var i = 0; i < text_arr.length; i++) {
      if(text_arr[i].trim().length > 0) { 
        settings_arr.push(text_arr[i]);
      }
    }
    return settings_arr;
  }


  d3.select("#order").on("change", function() {
      //throw('going to draw' + this.value);
      draw_graph(this.value);        
      check_iterations(this.value + '.txt');
      });

};

