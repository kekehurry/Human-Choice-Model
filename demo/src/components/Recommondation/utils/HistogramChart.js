import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';

export default function HistogramChart({ data, title, color = '#69b3a2' }) {
  const svgRef = useRef();
  const [width, setWidth] = useState(300);

  useEffect(() => {
    const handleResize = () => {
      if (svgRef.current) {
        setWidth(svgRef.current.parentElement.clientWidth);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (!data) return;

    const margin = { top: 20, right: 20, bottom: 20, left: 20 };
    const height = Math.min(width, 200);
    const svgWidth = width - margin.left - margin.right;
    const svgHeight = height - margin.top - margin.bottom;

    // Create the SVG element
    const svg = d3
      .select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('style', 'max-width: 100%; height: auto;');

    svg.selectAll('g').remove(); // Clear previous renderings

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create X and Y scales
    const x = d3
      .scaleLinear()
      .domain([0, d3.max(data, (d) => d.value)])
      .range([0, svgWidth]);

    const bins = d3.bin().domain(x.domain()).thresholds(x.ticks(10))(
      data.map((d) => d.value)
    );

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(bins, (d) => d.length)])
      .range([svgHeight, 0]);

    // Create bars
    g.selectAll('rect')
      .data(bins)
      .join('rect')
      .attr('x', (d) => x(d.x0))
      .attr('y', (d) => y(d.length))
      .attr('width', (d) => x(d.x1) - x(d.x0) - 1)
      .attr('height', (d) => svgHeight - y(d.length))
      .attr('fill', color);

    // Add X and Y axes
    g.append('g')
      .attr('transform', `translate(0,${svgHeight})`)
      .call(d3.axisBottom(x))
      .selectAll('text') // Change the color of the tick labels
      .attr('fill', 'white');

    // g.append('g').call(d3.axisLeft(y));
  }, [data, width]);

  return (
    <>
      <h2 className="text-md font-bold mt-8 ml-8 text-white ">{title}</h2>
      <div className="w-full h-full flex justify-center items-center">
        <svg ref={svgRef}></svg>
      </div>
    </>
  );
}
