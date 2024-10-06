import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const exampleData = [
  { name: 'F&B Eatery/Full-Service Restaurants', value: 52 },
  { name: 'F&B Eatery/Snack and Nonalcoholic Beverage Bars', value: 21 },
  { name: 'F&B Eatery/Drinking Places', value: 7 },
  { name: 'F&B Eatery/Limited-Service Restaurants', value: 19 },
];

const data = exampleData;

//settings
const width = 300;
const height = 100;
const marginTop = 30;
const marginRight = 15;
const marginBottom = 10;
const marginLeft = 15;
const metric = 'absolute';

export default function DivergingBarChart() {
  const svgRef = useRef();

  useEffect(() => {
    if (!data) return;
    // Create the positional scales.
    const barHeight =
      (height - marginTop - marginBottom) / Math.ceil(data.length + 0.1);

    const x = d3
      .scaleLinear()
      .domain(d3.extent(data, (d) => d.value))
      .rangeRound([marginLeft, width - marginRight]);

    const y = d3
      .scaleBand()
      .domain(data.map((d) => d.name))
      .rangeRound([marginTop, height - marginBottom])
      .padding(0.1);

    // Create the format function.
    const format = d3.format(metric === 'absolute' ? '+,d' : '+.1%');
    const tickFormat =
      metric === 'absolute' ? d3.formatPrefix('+.1', 1e6) : d3.format('+.0%');

    // Create the SVG container.
    const svg = d3
      .select(svgRef.current)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('style', 'max-width: 100%; height: auto; font: 10px sans-serif;');

    svg.selectAll('*').remove(); // Clear previous content

    // Add a rect for each name.
    svg
      .append('g')
      .selectAll('rect')
      .data(data)
      .join('rect')
      .attr('fill', (d) => d3.schemeRdBu[3][d.value > 0 ? 2 : 0])
      .attr('x', (d) => x(Math.min(d.value, 0)))
      .attr('y', (d) => y(d.name))
      .attr('width', (d) => Math.abs(x(d.value) - x(0)))
      .attr('height', y.bandwidth());

    // Add a text label for each name.
    svg
      .append('g')
      .attr('font-family', 'sans-serif')
      .attr('font-size', 10)
      .selectAll('text')
      .data(data)
      .join('text')
      .attr('text-anchor', (d) => (d.value < 0 ? 'end' : 'start'))
      .attr('x', (d) => x(d.value) + Math.sign(d.value - 0) * 4)
      .attr('y', (d) => y(d.name) + y.bandwidth() / 2)
      .attr('dy', '0.35em')
      .text((d) => format(d.value));

    // Add the axes and grid lines.
    svg
      .append('g')
      .attr('transform', `translate(0,${marginTop})`)
      .call(
        d3
          .axisTop(x)
          .ticks(width / 80)
          .tickFormat(tickFormat)
      )
      .call((g) =>
        g
          .selectAll('.tick line')
          .clone()
          .attr('y2', height - marginTop - marginBottom)
          .attr('stroke-opacity', 0.1)
      )
      .call((g) => g.select('.domain').remove());

    svg
      .append('g')
      .attr('transform', `translate(${x(0)},0)`)
      .call(d3.axisLeft(y).tickSize(0).tickPadding(6))
      .call((g) =>
        g
          .selectAll('.tick text')
          .filter((d, i) => data[i].value < 0)
          .attr('text-anchor', 'start')
          .attr('x', 6)
      );
  }, [data]);

  return <svg ref={svgRef}></svg>;
}
