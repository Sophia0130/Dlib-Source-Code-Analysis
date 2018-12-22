        //// apply_filters_to_fhog()����
        //// ���ܣ���HOG������������ĳһ���˲���31ά�˲���Ľ����ӣ�
        //// ������- saliency_image     ��ά�˲���Ӻ�Ľ����saliency_image��һ�����أ���Ӧ���������е�һ����ⴰ��
        ////       -feats    ĳһ������      - feats[i]     31ά�е�ĳһά
        ////       - area                �˲���δ��0���ľ�������
        
        template <typename fhog_filterbank>
        rectangle apply_filters_to_fhog (
            const fhog_filterbank& w,
            const array<array2d<float> >& feats,
            array2d<float>& saliency_image
        )
        {

#ifdef SHOW_FUNC_ENTRY
            std::cout << "apply_filters_to_fhog()" << endl;
#endif
            const unsigned long num_separable_filters = w.num_separable_filters();  
            rectangle area;

            //// ��31ά�е�ĳһά���˲����������������
            
            //// 1.�ڸ�������ʹ�� regular filter 
            // use the separable filters if they would be faster than running the regular filters.
            if (num_separable_filters > w.filters.size()*std::min(w.filters[0].nr(),w.filters[0].nc())/3.0)     //���㸴�ӶȵĹ�ʽ�ó�
            {
                area = spatially_filter_image(feats[0], saliency_image, w.filters[0]);                  //��һ���˲������д
                for (unsigned long i = 1; i < w.filters.size(); ++i)
                {
                    //// ��HOG��������ĳһ��ĳһά�ģ���31ά�е�ĳһά���˲����
                    // now we filter but the output adds to saliency_image rather than
                    // overwriting it.
                    spatially_filter_image(feats[i], saliency_image, w.filters[i], 1, false, true);     //��31ά���˲�������
                    
                }

#ifdef SHOW_FUNC_ENTRY
                std::cout << "Use Regular Filter" << endl;
                std::cout << "filters-size" << w.filters.size() << endl; //31ά
#endif
            }

            //// 2.�ڸ�������ʹ��seperable filter�ɷ����˲���
            else
            {
                saliency_image.clear();
                array2d<float> scratch;

                // find the first filter to apply
                unsigned long i = 0;
                while (i < w.row_filters.size() && w.row_filters[i].size() == 0) 
                    ++i;

                //// �ֳ�x��y��������������һά�˲���
                for (; i < w.row_filters.size(); ++i)
                {
                    for (unsigned long j = 0; j < w.row_filters[i].size(); ++j)
                    {
                        if (saliency_image.size() == 0)
                            area = float_spatially_filter_image_separable(feats[i], saliency_image, w.row_filters[i][j], w.col_filters[i][j],scratch,false);
                        else
                            area = float_spatially_filter_image_separable(feats[i], saliency_image, w.row_filters[i][j], w.col_filters[i][j],scratch,true);
                    }
                }
                if (saliency_image.size() == 0)
                {
                    saliency_image.set_size(feats[0].nr(), feats[0].nc());
                    assign_all_pixels(saliency_image, 0);
                }

#ifdef SHOW_FUNC_ENTRY
                std::cout << "Use Separable Filter" << endl;
                std::cout << "row col filter" << w.row_filters.size() << endl;
#endif
            }
            return area;
        }


        //// detect_from_fhog_pyramid()����
        //// ���ܣ�1.��HOG������������ĳһ���˲�����Ϊ31ά�˲������ӣ�
        ////       2.���ò�������ֵ���˲��������������ⴰ�ڣ���ԭ��ԭͼ
        ////       ע�⣺һ���˲������Ӧһ����ⴰ��
        //// ������- saliency_image   ����˲���Ľ��
        ////       - area             û�н���0���ľ�������
        ////       - filter_padding   

        template <
            typename pyramid_type,
            typename feature_extractor_type,
            typename fhog_filterbank
            >
        void detect_from_fhog_pyramid (
            const array<array<array2d<float> > >& feats,
            const feature_extractor_type& fe,
            const fhog_filterbank& w,
            const double thresh,
            const unsigned long det_box_height,
            const unsigned long det_box_width,
            const int cell_size,
            const int filter_rows_padding,
            const int filter_cols_padding,
            std::vector<std::pair<double, rectangle> >& dets
        ) 
        {
            dets.clear();

            array2d<float> saliency_image;
            pyramid_type pyr;

            //// ����ĳһ������
            // for all pyramid levels
            for (unsigned long l = 0; l < feats.size(); ++l)  
            {
                //// ÿһ�������˲���Ľ��
                const rectangle area = apply_filters_to_fhog(w, feats[l], saliency_image);  //��0��������

                //// �����˲�������ҳ�������ֵ���˲�������������ļ�ⴰ�ڻ�ԭ��ԭͼ����
                // now search the saliency image for any detections
                for (long r = area.top(); r <= area.bottom(); ++r)
                {
                    for (long c = area.left(); c <= area.right(); ++c)
                    {
                        // if we found a detection
                        if (saliency_image[r][c] >= thresh)
                        {
                            //// ����������������㻹ԭ��ԭͼ
                            rectangle rect = fe.feats_to_image(centered_rect(point(c,r),det_box_width,det_box_height), 
                                cell_size, filter_rows_padding, filter_cols_padding);
                            rect = pyr.rect_up(rect, l); 
                            dets.push_back(std::make_pair(saliency_image[r][c], rect));
                        }
                    }
                }
            }

#ifdef SHOW_FILTER_RES
            ////��l�������˲���Ľ��

            int level = 3;      //�ڼ���������
            std::cout << " HOG Pyramid Level: " << level << endl;
            const rectangle area = apply_filters_to_fhog(w, feats[level], saliency_image);  //��0��������
            for (long r = area.top(); r <= area.bottom(); ++r)
            {
                std::cout << " *  ";
                for (long c = area.left(); c <= area.right(); ++c)
                {
                    std::cout << fixed << setprecision(2) << saliency_image[r][c] << "  ";
                }
                std::cout << endl;
            }
            std::cout << endl << endl;
#endif

            std::sort(dets.rbegin(), dets.rend(), compare_pair_rect);   //�Ӵ�С����
        }


        //// evaluate_detectors()����
        //// ���ܣ�Ŀ����

        template <
            typename pyramid_type,
            typename image_type
            >
        void evaluate_detectors (
            const std::vector<object_detector<scan_fhog_pyramid<pyramid_type> > >& detectors,
            const image_type& img,
            std::vector<rect_detection>& dets,
            const double adjust_threshold = 0
        )
        {
            typedef scan_fhog_pyramid<pyramid_type> scanner_type;

            dets.clear();
            if (detectors.size() == 0)
                return;

            const unsigned long cell_size = detectors[0].get_scanner().get_cell_size();

            // Find the maximum sized filters and also most extreme pyramiding settings used.
            unsigned long max_filter_width = 0;
            unsigned long max_filter_height = 0;
            unsigned long min_pyramid_layer_width = std::numeric_limits<unsigned long>::max();
            unsigned long min_pyramid_layer_height = std::numeric_limits<unsigned long>::max();
            unsigned long max_pyramid_levels = 0;
            bool all_cell_sizes_the_same = true;

            //// ��ⴰ�ڵĳߴ磺ȡ����detector���
            //// ��С��������ĳߴ磺ȡ����detector��С
            //// �ж�ÿ��scanner��cell size��С�Ƿ���ͬ
            for (unsigned long i = 0; i < detectors.size(); ++i)
            {
                const scanner_type& scanner = detectors[i].get_scanner();
                max_filter_width = std::max(max_filter_width, scanner.get_fhog_window_width());      // ע�⣺����Ĵ������˲������ڲ������ش���
                max_filter_height = std::max(max_filter_height, scanner.get_fhog_window_height());   //
                max_pyramid_levels = std::max(max_pyramid_levels, scanner.get_max_pyramid_levels());
                min_pyramid_layer_width = std::min(min_pyramid_layer_width, scanner.get_min_pyramid_layer_width());
                min_pyramid_layer_height = std::min(min_pyramid_layer_height, scanner.get_min_pyramid_layer_height());
                if (cell_size != scanner.get_cell_size())
                    all_cell_sizes_the_same = false;
            }

    #ifdef SHOW_SINGLE_DETECT
            std::cout<<"max_filter_width "<<max_filter_width<<endl;   //ע�⣺����������������˲����ĳߴ磬���������ؼ��ĳߴ�
            std::cout<<"max_filter_height "<<max_filter_height<<endl;
            std::cout<<"all_cell_sizes_the_same "<<all_cell_sizes_the_same<<endl;
            max_filter_height��weightû��ʲôʵ�����ã����۶�hog������������ʲô�����˲������˲�����ĳߴ磬��hog�ĳߴ���ͬ
    #endif

            //// ����detector��cell�ߴ�һ�£�ֻҪ����һ��pyramid HOG
            std::vector<rect_detection> dets_accum;     //���򣺼����λ�á����Ŷȡ�ʹ�õ����ĸ�detector
            // Do to the HOG feature extraction to make the fhog pyramid.  Again, note that we
            // are making a pyramid that will work with any of the detectors.  But only if all
            // the cell sizes are the same.  If they aren't then we have to calculate the
            // pyramid for each detector individually.
            
            array<array<array2d<float> > > feats;       //hog����������

            //// ��ÿ��scanner��cell_size�ߴ�һ��������һ������������
            if (all_cell_sizes_the_same)
            {
                impl::create_fhog_pyramid<pyramid_type>(img,
                    detectors[0].get_scanner().get_feature_extractor(), feats, cell_size,
                    max_filter_height, max_filter_width, min_pyramid_layer_width,
                    min_pyramid_layer_height, max_pyramid_levels);
            }

            std::vector<std::pair<double, rectangle> > temp_dets;   //���ŶȺͼ�⵽��Ŀ���
            for (unsigned long i = 0; i < detectors.size(); ++i)    //��Ӧ�ڼ������Ƶ�detector[i]
            {
                //// ��scanner��cell size��ͬ��ÿ����������������
                const scanner_type& scanner = detectors[i].get_scanner();
                if (!all_cell_sizes_the_same)
                {
                    impl::create_fhog_pyramid<pyramid_type>(img,
                        scanner.get_feature_extractor(), feats, scanner.get_cell_size(),
                        max_filter_height, max_filter_width, min_pyramid_layer_width,
                        min_pyramid_layer_height, max_pyramid_levels);
                }
                
                //// ȷ������Ĵ�С
                const unsigned long det_box_width  = scanner.get_fhog_window_width()  - 2*scanner.get_padding();    
                const unsigned long det_box_height = scanner.get_fhog_window_height() - 2*scanner.get_padding();

    #ifdef SHOW_SINGLE_DETECT
                std::cout << "detector " << i+1 << " : " << "box_width " << det_box_width << endl;
                std::cout << "detector " << i+1 << " : " << "box_height " << det_box_height << endl << endl;
    #endif

                //// ʹ��һ��Ŀ���������õ����ϸü������Ŀ���
                // A single detector object might itself have multiple weight vectors in it. So
                // we need to evaluate all of them.
                for (unsigned d = 0; d < detectors[i].num_detectors(); ++d)     //ʹ�õĵڼ���detector[i]�еĵڼ����˲��� //detectors[i].num_detectors()=1 ��������ʹ�õĵ����˲���
                {
                    const double thresh = detectors[i].get_processed_w(d).w(scanner.get_num_dimensions());      //threshold ѵ����ֵ

    #ifdef SHOW_FILTER_RES
                    std::cout << " Gesture " << i+1 << " Detection " << endl;
    #endif
                    impl::detect_from_fhog_pyramid<pyramid_type>(feats, scanner.get_feature_extractor(),
                        detectors[i].get_processed_w(d).get_detect_argument(), thresh+adjust_threshold,         //thresh + adjust_threshold��adjust_threshold=-0.3
                        det_box_height, det_box_width, cell_size, max_filter_height,
                        max_filter_width, temp_dets);

                    for (unsigned long j = 0; j < temp_dets.size(); ++j)
                    {
                        rect_detection temp;
                        temp.detection_confidence = temp_dets[j].first-thresh;
                        temp.weight_index = i;
                        temp.rect = temp_dets[j].second;
                        dets_accum.push_back(temp);
                    }
                }
            }

            //// ͬһ��detector�õ������м���ʹ�÷Ǽ���ֵ���ƺϲ�
            // Do non-max suppression
            dets.clear();
            if (detectors.size() > 1)
                std::sort(dets_accum.rbegin(), dets_accum.rend());
            for (unsigned long i = 0; i < dets_accum.size(); ++i)
            {
                const test_box_overlap tester = detectors[dets_accum[i].weight_index].get_overlap_tester();     //ע�⣺tester �Ƕ�Ӧ��Ӧ��detector[i]��
                                                                                                                //tester��object_detector���е�һ����Ա����   boxes_overlap���ñ�����test_box_overlap�� 
                if (impl::overlaps_any_box(tester, dets, dets_accum[i]))
                    continue;

                dets.push_back(dets_accum[i]);
            }
        }
        
