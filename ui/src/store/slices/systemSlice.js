import { createSlice } from '@reduxjs/toolkit';

export const initializeSystem = () => (dispatch) => {};
export const checkSystemHealth = () => (dispatch) => {};

const slice = createSlice({
  name: 'system',
  initialState: { initialized: false, loading: false, error: null },
  reducers: {}
});

export const { actions } = slice;
export default slice;
