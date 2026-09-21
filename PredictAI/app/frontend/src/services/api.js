import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000/api",
  headers: {
    "Content-Type": "application/json",
  },
});

export const getHealth = async () => {
  const response = await API.get("/health");
  return response.data;
};

export const getSpacecrafts = async () => {
  const response = await API.get("/spacecrafts");
  return response.data;
};

export const getPredictions = async (channel) => {
  const response = await API.get(`/predictions/${channel}`);
  return response.data;
};

export const getDiagnosis = async (channel) => {
  const response = await API.get(`/diagnosis/${channel}`);
  return response.data;
};

export const getChannels = async () => {
  const response = await API.get("/channels");
  return response.data;
};

export default API;
