'use client';
import GraphViewer from '@/components/GraphViewer';
import SearchBar from '@/components/SearchBar';
import Recommondation from '@/components/Recommondation';
import DataProvider from '@/context/provider';

export default function Home() {
  return (
    <div>
      <DataProvider>
        <div className="flex">
          <div className="w-1/4 h-screen flex-row">
            <Recommondation />
          </div>
          <div className="w-3/4 h-screen flex-row">
            <SearchBar />
            <GraphViewer />
          </div>
        </div>
      </DataProvider>
    </div>
  );
}
