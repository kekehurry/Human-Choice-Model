'use client';
import { ForceGraph3D } from 'react-force-graph';
import { useRef, useState, useEffect, useContext } from 'react';
import { StateContext } from '@/context/provider';

export default function GraphViewer() {
  const ref = useRef();
  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);
  const [gData, setGData] = useState({ nodes: [], links: [] });
  const { state, setState } = useContext(StateContext);

  useEffect(() => {
    const handleResize = () => {
      const container = document.getElementById('graph');
      if (container) {
        setWidth(container.clientWidth);
        setHeight(container.clientHeight);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (!state.graphData) return;
    setGData(state.graphData);
  }, [state.graphData]);

  return (
    <div id="graph" className="w-full h-full">
      <ForceGraph3D
        graphData={gData}
        ref={ref}
        width={width}
        height={height}
        nodeAutoColorBy={'type'}
        nodeRelSize={6}
      />
    </div>
  );
}
