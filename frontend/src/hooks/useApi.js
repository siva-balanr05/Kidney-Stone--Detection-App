import { useState, useEffect, useCallback } from 'react';

/**
 * Custom hook for API calls with loading, error, and data state management.
 * @param {Function} apiCall - async function that returns data
 * @param {Array} deps - dependency array (like useEffect)
 * @param {boolean} immediate - if true, call immediately on mount
 */
export function useApi(apiCall, deps = [], immediate = true) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(immediate);
  const [error, setError] = useState(null);

  const execute = useCallback(async (...args) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiCall(...args);
      setData(result);
      return result;
    } catch (err) {
      setError(err.message || 'An error occurred');
      return null;
    } finally {
      setLoading(false);
    }
  }, deps); // eslint-disable-line

  useEffect(() => {
    if (immediate) execute();
  }, [immediate, ...deps]); // eslint-disable-line

  return { data, loading, error, execute, setData };
}

/**
 * Hook for mutation-style API calls (no auto-execute).
 */
export function useMutation(apiCall) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const mutate = useCallback(async (...args) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiCall(...args);
      return { data: result, error: null };
    } catch (err) {
      const msg = err.message || 'An error occurred';
      setError(msg);
      return { data: null, error: msg };
    } finally {
      setLoading(false);
    }
  }, []); // eslint-disable-line

  return { mutate, loading, error };
}

export default useApi;
