import { useEffect, useState } from 'react';
import Link from 'next/link';

const BlogIndex = () => {
  const [posts, setPosts] = useState([]);

  useEffect(() => {
    const fetchPosts = async () => {
      const response = await fetch('/api/posts');
      const data = await response.json();
      setPosts(data);
    };

    fetchPosts();
  }, []);

  return (
    <div>
      <h1>Blog Posts</h1>
      <ul>
        {posts.map((post) => (
          <li key={post.id}>
            <Link href={`/blog/${post.id}`}>
              <a>
                <h2>{post.title}</h2>
                <p>{post.excerpt}</p>
              </a>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default BlogIndex;