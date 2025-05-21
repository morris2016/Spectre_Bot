import { createSlice } from '@reduxjs/toolkit';

export const initializePreferences = () => (dispatch) => {};

const slice = createSlice({
  name: 'preferences',
  initialState: {},
  reducers: {}
});

export const { actions } = slice;
export default slice;
