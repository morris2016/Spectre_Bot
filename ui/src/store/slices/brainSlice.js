import { createSlice } from '@reduxjs/toolkit';

const slice = createSlice({
  name: 'brain',
  initialState: {
    status: 'idle',
    activeBrainId: null
  },
  reducers: {
    setStatus(state, action) {
      state.status = action.payload;
    },
    setActiveBrain(state, action) {
      state.activeBrainId = action.payload;
    }
  }
});

export const { actions } = slice;
export default slice;
