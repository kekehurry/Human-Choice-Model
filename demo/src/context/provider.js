import React, { createContext, useState } from 'react';

const initState = {
  graphData: null,
  amenity_choices: null,
  mobility_choices: null,
  ages: null,
  incomes: null,
};

export const StateContext = createContext();

export default function DataProvider({ children }) {
  const [state, setState] = useState(initState);

  return (
    <StateContext.Provider value={{ state, setState }}>
      {children}
    </StateContext.Provider>
  );
}
