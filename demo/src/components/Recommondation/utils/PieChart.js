import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';

export default function Chart({ data, title, choice }) {
  const svgRef = useRef();
  const [width, setWidth] = useState(300);

  useEffect(() => {
    const handleResize = () => {
      if (svgRef.current) {
        setWidth(svgRef.current.parentElement.clientWidth * 0.8);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (!data) return;

    const height = Math.min(width, 500);
    const radius = Math.min(width, height) / 2;

    const arc = d3
      .arc()
      .innerRadius(radius * 0.67)
      .outerRadius(radius - 2);

    const arcHover = d3
      .arc()
      .innerRadius(radius * 0.6)
      .outerRadius(radius - 1); // Slightly larger for hover effect

    const pie = d3
      .pie()
      .padAngle(1 / radius)
      .sort(null)
      .value((d) => d.value);

    const color = d3
      .scaleOrdinal()
      .domain(data.map((d) => d.name))
      .range(
        d3
          .quantize((t) => d3.interpolateSpectral(t * 0.8 + 0.1), data.length)
          .reverse()
      );

    const svg = d3
      .select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [-width / 2, -height / 2, width, height])
      .attr('style', 'max-width: 100%; height: auto;');

    svg.selectAll('g').remove(); // Clear previous renderings

    svg
      .append('g')
      .selectAll('path')
      .data(pie(data))
      .join('path')
      .attr('fill', (d) => color(d.data.name))
      .attr('d', arc)
      .on('mouseover', function (event, d) {
        d3.select(this).transition().duration(200).attr('d', arcHover); // Enlarge the arc
      })
      .on('mouseout', function (event, d) {
        d3.select(this).transition().duration(200).attr('d', arc); // Return to normal size
      })
      .append('title')
      .text((d) => `${d.data.name}: ${d.data.value.toLocaleString()}`);

    svg
      .append('circle')
      .attr('cx', 0) // Center the circle
      .attr('cy', 0) // Center the circle
      .attr('r', radius * 0.3) // Set radius smaller than inner radius of the arcs
      .attr('fill', choice ? color(choice) : '#000000')
      .on('mouseover', function (event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', radius * 0.5);
      })
      .on('mouseout', function (event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', radius * 0.3);
      })
      .append('title')
      .text(choice);

    if (choice) {
      svg
        .append('text')
        .attr('x', 0)
        .attr('y', 0)
        .attr('text-anchor', 'middle') // Center text horizontally
        .attr('dominant-baseline', 'middle') // Center text vertically
        .attr('fill', '#000000') // Text color (adjust as needed)
        .attr('font-size', '8px')
        .text('LLM Choice');
    }
  }, [data, width]);

  return (
    <>
      <h2 className="text-md font-bold mt-2 text-white text-center">{title}</h2>
      <div className="w-full h-full flex justify-center items-center">
        <svg ref={svgRef}></svg>
      </div>
    </>
  );
}
