import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  Paper,
  List,
  ListItem,
  ListItemText,
  Link,
  FormControl,
  InputLabel,
  Select,
  MenuItem,

} from '@mui/material';
import { api } from '../../api';

const NewsFeed = ({ limit = 20 }) => {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [category, setCategory] = useState('all');

  const categories = [
    { value: 'all', label: 'All' },
    { value: 'crypto', label: 'Crypto' },
    { value: 'forex', label: 'Forex' },
    { value: 'stocks', label: 'Stocks' },
  ];


  useEffect(() => {
    const loadNews = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data } = await api.market.getMarketNews({ limit, category });

        setNews(data || []);
      } catch (err) {
        console.error('Failed to load market news', err);
        setError('Failed to load market news');
      } finally {
        setLoading(false);
      }
    };

    loadNews();
  }, [limit, category]);

  

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Typography color="error" variant="body2">
        {error}
      </Typography>
    );
  }

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" sx={{ mb: 1 }}>
        Latest News
      </Typography>
      <FormControl size="small" sx={{ mb: 2, minWidth: 150 }}>
        <InputLabel id="news-category-label">Category</InputLabel>
        <Select
          labelId="news-category-label"
          value={category}
          label="Category"
          onChange={(e) => setCategory(e.target.value)}
        >
          {categories.map((c) => (
            <MenuItem value={c.value} key={c.value}>
              {c.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <List dense>
        {news.map((item) => (
          <ListItem key={item.id} divider>
            <ListItemText
              primary={
                <Link href={item.url} target="_blank" rel="noopener noreferrer">
                  {item.title}
                </Link>
              }
              secondary={new Date(item.publishedAt).toLocaleString()}
            />
          </ListItem>
        ))}
        {news.length === 0 && (
          <ListItem>
            <ListItemText primary="No news available" />
          </ListItem>
        )}
      </List>
    </Paper>
  );
};

export default NewsFeed;
