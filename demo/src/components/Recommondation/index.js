import BarChart from './utils/BarChart';
import PieChart from './utils/PieChart';
import HistogramChart from './utils/HistogramChart';
import { StateContext } from '@/context/provider';
import { useContext, useEffect } from 'react';

const exampleData = [
  { name: 'F&B Eatery/Full-Service Restaurants', value: 52 },
  { name: 'F&B Eatery/Snack and Nonalcoholic Beverage Bars', value: 21 },
  { name: 'F&B Eatery/Drinking Places', value: 7 },
  { name: 'F&B Eatery/Limited-Service Restaurants', value: 19 },
];

const histData = [
  { name: 'A', value: 5 },
  { name: 'B', value: 5 },
  { name: 'C', value: 15 },
  { name: 'D', value: 20 },
  { name: 'E', value: 25 },
  { name: 'F', value: 30 },
  { name: 'G', value: 35 },
  { name: 'H', value: 40 },
  { name: 'I', value: 90 },
  { name: 'J', value: 100 },
];

export default function Recommondation() {
  const { state, setState } = useContext(StateContext);
  return (
    <div className="w-full h-full bg-gray-950">
      <div className="w-full flex flex-row space-x-2 pt-16 pb-4 justify-center">
        <div className="w-1/2">
          <PieChart
            data={state.amenity_choices}
            title="Amenity Choice"
            choice={state.amenity_final_choice}
          />
        </div>
        <div className="w-1/2">
          <PieChart
            data={state.mobility_choices}
            title="Mobility Choice"
            choice={state.mobility_final_choice}
          />
        </div>
      </div>
      <div className="w-full">
        <HistogramChart data={state.ages} title="Age Range" color="#69b3a2" />
      </div>
      <div className="w-full">
        <HistogramChart
          data={state.incomes}
          title="Income Range"
          color="#69b3a2"
        />
      </div>
    </div>
  );
}
