'use client';
import { useState, useContext, useEffect } from 'react';
import { StateContext } from '@/context/provider';

export default function SearchBar() {
  const [formData, setFormData] = useState({
    profile: 'a young adult with high income',
    desire: 'Eat',
  });
  const { state, setState } = useContext(StateContext);
  const [searching, setSearching] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('');

  const handleFormChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  };

  const handleSearch = (e) => {
    setSearching(true);
    fetch('http://127.0.0.1:5005/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profile: formData.profile,
        desire: formData.desire,
        k: 50,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        setState({
          ...state,
          graphData: data['graph'],
          amenity_choices: data['amenity_choices'],
          mobility_choices: data['mobility_choices'],
          amenity_final_choice: data['amenity_final_choice'],
          mobility_final_choice: data['mobility_final_choice'],
          ages: data['ages'],
          incomes: data['incomes'],
        });
      })
      .catch((err) => console.error(err))
      .finally(() => setSearching(false));
  };

  const handleReset = (e) => {
    setLoading(true);
    fetch('http://127.0.0.1:5005/init')
      .then(async (res) => {
        return await res.json();
      })
      .then((data) => {
        setState({
          ...state,
          graphData: data['graph'],
          amenity_choices: data['amenity_choices'],
          mobility_choices: data['mobility_choices'],
          ages: data['ages'],
          incomes: data['incomes'],
        });
      })
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  };

  const textLoading = [
    'searching for similar profiles...',
    'createing behavior graph...',
    'getting LLM responses...',
  ];

  useEffect(() => {
    handleReset();
  }, []);

  useEffect(() => {
    let textIndex = 0;
    if (searching) {
      const intervalId = setInterval(() => {
        setLoadingText(textLoading[textIndex]);
        textIndex = (textIndex + 1) % textLoading.length;
      }, 2000); // Change text every 2 seconds
      return () => clearInterval(intervalId); // Clean up on unmount or loading = false
    } else {
      setLoadingText(''); // Reset the text when loading is false
    }
  }, [searching]);

  return (
    <div className="absolute top-10 left-2/3 transform -translate-x-2/3 z-10 ">
      <h1 className="text-3xl font-bold text-white text-center pt-4">
        Humanised Choice Model
      </h1>
      <div className="flex flex-row space-x-2 items-end mt-8">
        <div>
          <input
            type="text"
            id="profile"
            className="mt-1 block px-2 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm text-gray-800 w-80"
            style={{ height: 40 }}
            value={formData.profile}
            onChange={handleFormChange}
          />
        </div>
        <div>
          <select
            id="desire"
            className="mt-1 block px-2 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm text-gray-800"
            style={{ height: 40 }}
            value={formData.desire}
            onChange={handleFormChange}
          >
            <option value="Eat">Eat</option>
            <option value="Shop">Shop</option>
            <option value="Recreation">Recreation</option>
          </select>
        </div>

        <div className="flex flex-row space-x-2 items-end">
          <button
            className="bg-blue-500 hover:bg-blue-700 text-white mt-1 block px-2 py-2 rounded-md shadow-sm text-sm"
            style={{ width: 100, height: 40 }}
            onClick={handleSearch}
          >
            {searching && <i className="fa fa-spinner fa-spin"></i>} Search
          </button>
          <button
            className="bg-blue-500 hover:bg-blue-700 text-white mt-1 block px-2 py-2 rounded-md shadow-sm text-sm"
            style={{ width: 100, height: 40 }}
            onClick={handleReset}
          >
            {loading && <i className="fa fa-spinner fa-spin"></i>} Reset
          </button>
        </div>
      </div>
      <p className="text-white text-center text-sm mt-10">{loadingText}</p>
    </div>
  );
}
